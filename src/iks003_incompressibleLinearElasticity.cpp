// SPDX-FileCopyrightText: 2021-2024 The Ikarus Developers mueller@ibb.uni-stuttgart.de
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <config.h>

#include <matplot/matplot.h>

#include <dune/alugrid/grid.hh>
#include <dune/foamgrid/foamgrid.hh>
#include <dune/functions/functionspacebases/boundarydofs.hh>
#include <dune/functions/functionspacebases/compositebasis.hh>
#include <dune/functions/functionspacebases/lagrangebasis.hh>
#include <dune/functions/functionspacebases/powerbasis.hh>
#include <dune/functions/functionspacebases/subspacebasis.hh>
#include <dune/functions/gridfunctions/discreteglobalbasisfunction.hh>
#include <dune/grid/io/file/vtk/vtkwriter.hh>
#include <dune/localfefunctions/cachedlocalBasis/cachedlocalBasis.hh>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <autodiff/forward/dual/dual.hpp>

#include <ikarus/assembler/simpleassemblers.hh>
#include <ikarus/finiteelements/febases/autodifffe.hh>
#include <ikarus/finiteelements/fetraits.hh>
#include <ikarus/finiteelements/physicshelper.hh>
#include <ikarus/utils/algorithms.hh>
#include <ikarus/utils/basis.hh>
#include <ikarus/utils/dirichletvalues.hh>
#include <ikarus/utils/drawing/griddrawer.hh>
#include <ikarus/utils/init.hh>

using namespace Ikarus;
using namespace Dune::Indices;
template <typename Basis_, typename FERequirements_ = FErequirements<>, bool useEigenRef = false>
struct Solid {
 public:
  using Traits            = FETraits<Basis_, FERequirements_, useEigenRef>;
  using Basis             = typename Traits::Basis;
  using FlatBasis         = typename Traits::FlatBasis;
  using FERequirementType = typename Traits::FERequirementType;
  using LocalView         = typename Traits::LocalView;
  using Geometry          = typename Traits::Geometry;
  using Element           = typename Traits::Element;

  Solid(const Basis &basis, const typename LocalView::Element &element, double emod, double nu)
      : localView_{basis.flat().localView()}, emod_{emod}, nu_{nu} {
    localView_.bind(element);
    mu_       = emod_ / (2 * (1 + nu_));
    lambdaMat = convertLameConstants({.emodul = emod_, .nu = nu_}).toLamesFirstParameter();
  }

  using GlobalIndex = typename LocalView::MultiIndex;
  void globalFlatIndices(std::vector<GlobalIndex> &globalIndices) const {
    const auto &feDisp = localView_.tree().child(_0, 0).finiteElement();
    for (size_t i = 0; i < feDisp.size(); ++i) {
      for (int j = 0; j < Traits::worlddim; ++j) {
        globalIndices.push_back(localView_.index((localView_.tree().child(_0, j).localIndex(i))));
      }
    }
    const auto &fePressure = localView_.tree().child(_1).finiteElement();
    for (size_t i = 0; i < fePressure.size(); ++i) {
      globalIndices.push_back(localView_.index((localView_.tree().child(_1).localIndex(i))));
    }
  }

  inline double calculateScalar(const FERequirementType &par) const { return calculateScalarImpl<double>(par); }

  [[nodiscard]] constexpr int size() const { return localView_.size(); }
  const Element &gridElement() { return localView_.element(); }
  const LocalView &localView() const { return localView_; }
  LocalView &localView() { return localView_; }

 protected:
  template <class ScalarType>
  auto calculateScalarImpl(const FERequirementType &par, const std::optional<const Eigen::VectorX<ScalarType>> &dx
                                                         = std::nullopt) const -> ScalarType {
    const auto &d      = par.getGlobalSolution(Ikarus::FESolutions::displacement);
    const auto &lambda = par.getParameter(Ikarus::FEParameter::loadfactor);
    Eigen::VectorX<ScalarType> localDisp(localView_.size());
    localDisp.setZero();
    auto &displacementNode = localView_.tree().child(_0, 0);
    auto &pressureNode     = localView_.tree().child(_1);
    const auto &feDisp     = displacementNode.finiteElement();
    const auto &fePressure = pressureNode.finiteElement();
    Eigen::Matrix<ScalarType, Traits::dimension, Eigen::Dynamic> disp;
    disp.setZero(Eigen::NoChange, feDisp.size());
    Eigen::Vector<ScalarType, Eigen::Dynamic> pN;
    pN.setZero(fePressure.size());

    if (dx) {
      for (auto i = 0U; i < feDisp.size(); ++i)
        for (auto k2 = 0U; k2 < Traits::mydim; ++k2)
          disp.col(i)(k2) = dx.value()[i * Traits::mydim + k2]
                            + d[localView_.index(localView_.tree().child(_0, k2).localIndex(i))[0]];
      for (auto i = 0U; i < fePressure.size(); ++i)
        pN[i] = dx.value()[Traits::mydim * feDisp.size() + i]
                + d[localView_.index(localView_.tree().child(_1).localIndex(i))[0]];
    } else {
      for (auto i = 0U; i < feDisp.size(); ++i)
        for (auto k2 = 0U; k2 < Traits::mydim; ++k2)
          disp.col(i)(k2) = d[localView_.index(localView_.tree().child(_0, k2).localIndex(i))[0]];
      for (auto i = 0U; i < fePressure.size(); ++i)
        pN[i] = d[localView_.index(localView_.tree().child(_1).localIndex(i))[0]];
    }

    ScalarType energy = 0.0;

    const int order  = 2 * (feDisp.localBasis().order());
    const auto &rule = Dune::QuadratureRules<double, Traits::mydim>::rule(localView_.element().type(), order);
    const auto geo   = localView_.element().geometry();
    Dune::CachedLocalBasis localBasisDisp(feDisp.localBasis());
    Dune::CachedLocalBasis localBasisPressure(fePressure.localBasis());
    Eigen::Matrix<double, Eigen::Dynamic, Traits::mydim> dNdisp;
    Eigen::VectorXd Ndisp;
    Eigen::VectorXd Npressure;
    for (auto &&gp : rule) {
      const auto J = Dune::toEigen(geo.jacobianTransposed(gp.position())).transpose().eval();
      localBasisDisp.evaluateFunctionAndJacobian(gp.position(), Ndisp, dNdisp);
      localBasisPressure.evaluateFunction(gp.position(), Npressure);
      const Eigen::Vector<double, Traits::worlddim> X = Dune::toEigen(geo.global(gp.position()));
      Eigen::Vector<ScalarType, Traits::worlddim> x   = X;
      for (auto i = 0U; i < feDisp.size(); ++i)
        x += disp.col(i) * Ndisp[i];

      ScalarType pressure = pN.dot(Npressure);

      const auto gradu      = (disp * dNdisp.template cast<ScalarType>() * J.inverse()).eval();
      const auto symgradu   = Dune::sym(gradu);
      const ScalarType divU = gradu.trace();

      Eigen::Vector<double, Traits::worlddim> fext;
      fext.setZero();
      fext[1] = lambda;
      fext[0] = 0 * lambda;

      energy += (0.5 * (2 * mu_ * symgradu.squaredNorm() - 1 / lambdaMat * Dune::power(pressure, 2)) + pressure * divU
                 - x.dot(fext))
                * geo.integrationElement(gp.position()) * gp.weight();  // plane strain for 2D
    }
    return energy;
  }

 private:
  LocalView localView_;
  double emod_;
  double nu_;
  double mu_;
  double lambdaMat;
};

int main(int argc, char **argv) {
  using namespace Dune::Functions;
  /// Construct grid
  Ikarus::init(argc, argv);
  using namespace Ikarus;
  using namespace Dune::Indices;
  constexpr int gridDim = 2;

  using Grid        = Dune::YaspGrid<gridDim>;
  const double L    = 1;
  const double h    = 1;
  const size_t elex = 20;
  const size_t eley = 20;

  Dune::FieldVector<double, 2> bbox       = {L, h};
  std::array<int, 2> elementsPerDirection = {elex, eley};
  auto grid                               = std::make_shared<Grid>(bbox, elementsPerDirection);
  auto gridView                           = grid->leafGridView();
  //  draw(gridView);

  using namespace Dune::Functions::BasisFactory;
  /// Construct basis
  auto basis = Ikarus::makeBasis(gridView, composite(power<2>(lagrange<1>()), lagrange<0>()));

  /// Create finite elements
  const double Emod = 2.1e1;
  const double nu   = 0.5;
  std::vector<AutoDiffFE<Solid<decltype(basis)>>> fes;
  for (auto &ele : elements(gridView))
    fes.emplace_back(basis, ele, Emod, nu);

  /// Collect dirichlet nodes
  auto basisP = std::make_shared<const decltype(basis)>(basis);
  Ikarus::DirichletValues dirichletValues(basisP->flat());

  dirichletValues.fixDOFs([](auto &basis_, auto &dirichletFlags) {
    Dune::Functions::forEachBoundaryDOF(subspaceBasis(basis_, _0),
                                        [&](auto &&localIndex, auto &&localView, auto &&intersection) {
                                          if (std::abs(intersection.geometry().center()[1]) < 1e-8)
                                            dirichletFlags[localView.index(localIndex)] = true;
                                        });
  });

  /// Create assembler
  auto sparseFlatAssembler = SparseFlatAssembler(fes, dirichletValues);

  /// Create function for external forces and stiffness matrix
  double lambda = 0;
  Eigen::VectorXd d;
  d.setZero(basis.flat().size());

  auto req = FErequirements().addAffordance(Ikarus::AffordanceCollections::elastoStatics);

  auto fextFunction = [&](auto &&lambdaLocal, auto &&dLocal) -> auto & {
    req.insertGlobalSolution(Ikarus::FESolutions::displacement, dLocal)
        .insertParameter(Ikarus::FEParameter::loadfactor, lambdaLocal);
    return sparseFlatAssembler.getReducedVector(req);
  };
  auto KFunction = [&](auto &&lambdaLocal, auto &&dLocal) -> auto & {
    req.insertGlobalSolution(Ikarus::FESolutions::displacement, dLocal)
        .insertParameter(Ikarus::FEParameter::loadfactor, lambdaLocal);
    return sparseFlatAssembler.getReducedMatrix(req);
  };

  auto K = KFunction(1.0, d);
  auto R = fextFunction(1.0, d);
  Eigen::SparseLU<decltype(K)> ld;
  ld.compute(K);
  if (ld.info() != Eigen::Success) DUNE_THROW(Dune::MathError, "Failed Compute");

  d -= sparseFlatAssembler.createFullVector(ld.solve(R));
  if (ld.info() != Eigen::Success) DUNE_THROW(Dune::MathError, "Failed Solve");

  /// Postprocess
  auto disp = Dune::Functions::makeDiscreteGlobalBasisFunction<Dune::FieldVector<double, 2>>(
      subspaceBasis(basis.flat(), _0), d);
  auto pressure = Dune::Functions::makeDiscreteGlobalBasisFunction<double>(subspaceBasis(basis.flat(), _1), d);
  Dune::VTKWriter vtkWriter(gridView, Dune::VTK::nonconforming);
  vtkWriter.addVertexData(disp, Dune::VTK::FieldInfo("displacement", Dune::VTK::FieldInfo::Type::vector, 2));
  vtkWriter.addVertexData(pressure, Dune::VTK::FieldInfo("pressure", Dune::VTK::FieldInfo::Type::scalar, 1));
  vtkWriter.write("iks003_incompressibleLinearElasticity");
}
