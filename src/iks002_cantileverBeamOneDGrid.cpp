// SPDX-FileCopyrightText: 2022 The Ikarus Developers mueller@ibb.uni-stuttgart.de
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <config.h>

#include <dune/common/exceptions.hh>
#include <dune/common/float_cmp.hh>
#include <dune/common/indices.hh>
#include <dune/functions/functionspacebases/basistags.hh>
#include <dune/functions/functionspacebases/boundarydofs.hh>
#include <dune/functions/functionspacebases/compositebasis.hh>
#include <dune/functions/functionspacebases/lagrangebasis.hh>
#include <dune/functions/functionspacebases/subspacebasis.hh>
#include <dune/functions/gridfunctions/analyticgridviewfunction.hh>
#include <dune/functions/gridfunctions/discreteglobalbasisfunction.hh>
#include <dune/geometry/quadraturerules.hh>
#include <dune/grid/onedgrid.hh>
#include <dune/localfefunctions/cachedlocalBasis/cachedlocalBasis.hh>

#include <Eigen/Core>

#include <ikarus/solver/linearSolver/linearSolver.hh>
#include <ikarus/utils/basis.hh>
#include <ikarus/utils/drawing/griddrawer.hh>
#include <ikarus/utils/init.hh>
using namespace Dune::Functions::BasisFactory;
using namespace Dune::Functions;
using namespace Dune::Functions::BasisBuilder;
using namespace Dune::Indices;

void TimoshenkoBeamStiffness(auto &KLocal, auto &localView, auto &gridElement, auto &quadratureRule,
                             const Eigen::Matrix2d &C) {
  using namespace Dune::Indices;

  auto wFE   = localView.tree().child(_0);
  auto phiFE = localView.tree().child(_1);
  Dune::CachedLocalBasis basisW(wFE.finiteElement().localBasis());
  Dune::CachedLocalBasis basisPhi(phiFE.finiteElement().localBasis());

  /// Determinant of Jacobian, obtained from gridElement
  auto detJ = gridElement.geometry().volume();

  /// get number of DOFs for w and phi
  auto numDofsW      = wFE.size();
  auto numDofsPhi    = phiFE.size();
  auto numDofsPerEle = numDofsW + numDofsPhi;

  /// initialize quantities
  KLocal.setZero(numDofsPerEle, numDofsPerEle);
  Eigen::VectorXd dNwDxi   = Eigen::VectorXd::Zero(numDofsW);
  Eigen::VectorXd Nphi     = Eigen::VectorXd::Zero(numDofsPhi);
  Eigen::VectorXd dNphiDxi = Eigen::VectorXd::Zero(numDofsPhi);
  Eigen::Matrix2Xd B;

  /// integration point loop
  for (auto &gp : quadratureRule) {
    // evaluate ansatz functions and their derivatives
    basisW.evaluateJacobian(gp.position(), dNwDxi);
    basisPhi.evaluateFunction(gp.position(), Nphi);
    basisPhi.evaluateJacobian(gp.position(), dNphiDxi);

    /// setup B-operator
    B.setZero(Eigen::NoChange, numDofsPerEle);

    /// fill columns of B-Operator related to w-DOFs
    for (unsigned int i = 0; i < wFE.size(); ++i)
      B(1, wFE.localIndex(i)) = dNwDxi[i] / detJ;

    /// fill columns of B-Operator related to phi-DOFs
    for (unsigned int i = 0; i < phiFE.size(); ++i)
      B.col(phiFE.localIndex(i)) << dNphiDxi[i] / detJ, Nphi[i];

    /// integration of stiffness matrix
    KLocal += B.transpose() * C * B * detJ * gp.weight();
  }
}

enum class TimoshenkoBeam { w, phi };

unsigned int getGlobalDofIdImpl(const auto &basis, const double position) {
  auto localView       = basis.localView();
  auto seDOFs          = subEntityDOFs(basis);
  const auto &gridView = basis.gridView();
  for (auto &element : elements(gridView)) {
    localView.bind(element);
    for (size_t i = 0U; i < element.subEntities(1); ++i) {
      if (Dune::FloatCmp::eq(element.template subEntity<1>(i).geometry().center()[0], position, 1e-8)) {
        auto &localIndex = seDOFs.bind(localView, i, 1);
        assert(localIndex.size() == 1 && "It is expected that only one w-DOF is associated with a vertex");
        return localView.index(localIndex[0])[0];
      }
    }
  }
  DUNE_THROW(Dune::InvalidStateException,
             "There is no desired DOF at the requested position. Currently, only "
             "DOFs at vertices are supported.");
}

unsigned int getGlobalDofId(TimoshenkoBeam requestedQuantity, const auto &basis, const double position) {
  using namespace Dune::Indices;
  if (requestedQuantity == TimoshenkoBeam::w)
    return getGlobalDofIdImpl(subspaceBasis(basis.flat(), _0), position);
  else if (requestedQuantity == TimoshenkoBeam::phi)
    return getGlobalDofIdImpl(subspaceBasis(basis.flat(), _1), position);
  else
    DUNE_THROW(Dune::InvalidStateException, "The requested quantity is not supported");
}

void plotDeformedTimoschenkoBeam(auto &gridView, auto &basis, auto &d_glob, double EI, double GA, double L, double F) {
  using namespace Dune::Indices;
  auto wGlobal   = makeDiscreteGlobalBasisFunction<double>(subspaceBasis(basis.flat(), _0), d_glob);
  auto phiGlobal = makeDiscreteGlobalBasisFunction<double>(subspaceBasis(basis.flat(), _1), d_glob);

  auto wSol = [&](auto x) {
    return -F * Dune::power(x[0], 3) / (6.0 * EI) + L * F * Dune::power(x[0], 2) / (2.0 * EI) + F * x[0] / GA;
  };
  auto phiSol            = [&](auto x) { return F * Dune::power(x[0], 2) / (2.0 * EI) - L * F * x[0] / EI; };
  auto wGlobalAnalytic   = makeAnalyticGridViewFunction(wSol, gridView);
  auto phiGlobalAnalytic = makeAnalyticGridViewFunction(phiSol, gridView);

  using namespace matplot;
  auto f = figure(true);
  tiledlayout(1, 2);
  auto ax1 = nexttile();
  auto ax2 = nexttile();
  hold(ax1, true);
  hold(ax2, true);

  auto localView        = basis.flat().localView();
  auto wLocal           = localFunction(wGlobal);
  auto phiLocal         = localFunction(phiGlobal);
  auto wLocalAnalytic   = localFunction(wGlobalAnalytic);
  auto phiLocalAnalytic = localFunction(phiGlobalAnalytic);
  std::vector<double> x = linspace(0, 1, 10);
  std::vector<double> x_L;
  std::vector<double> yw, yphi, ywAna, yphiAna;
  for (auto &edge : elements(gridView)) {
    wLocal.bind(edge);
    wLocalAnalytic.bind(edge);
    phiLocal.bind(edge);
    phiLocalAnalytic.bind(edge);
    localView.bind(edge);
    x_L     = transform(x, [&](auto x) { return edge.geometry().global({x}); });
    yw      = transform(x, [&](auto x) { return wLocal({x}); });
    ywAna   = transform(x, [&](auto x) { return wLocalAnalytic({x}); });
    yphi    = transform(x, [&](auto x) { return phiLocal({x}); });
    yphiAna = transform(x, [&](auto x) { return phiLocalAnalytic({x}); });

    auto l0 = ax1->plot(x_L, yw);
    l0->line_width(2);
    l0->color("blue");

    auto l0_ana = ax1->plot(x_L, ywAna);
    l0_ana->line_width(2);
    l0_ana->color("red");

    auto l1 = ax2->plot(x_L, yphi);
    l1->line_width(2);
    l1->color("blue");

    auto l1_ana = ax2->plot(x_L, yphiAna);
    l1_ana->line_width(2);
    l1_ana->color("red");
  }

  //  f->draw();
  //  using namespace std::chrono_literals;
  //  std::this_thread::sleep_for(5s);
}

void exampleTimoshenkoBeam(const int polynomialOrderW, const int polynomialOrderPhi, const int numElements) {
  const double b  = 1;
  const double L  = 10;
  const double E  = 1000;
  const double G  = E / 2;  // Poisson's ratio = 0
  const double t  = 1e-3;
  const double EI = E * b * t * t * t / 12.0;
  const double GA = G * b * t;
  const double F  = 1 * t * t * t;
  Eigen::Matrix2d C;
  C << EI, 0, 0, GA;
  const int maxOrderIntegration = std::max(2 * (polynomialOrderW - 1), 2 * polynomialOrderPhi);
  Dune::OneDGrid grid(numElements, 0, L);
  auto gridView = grid.leafGridView();
  // draw(gridView);

  /// Basis  with different orders for w (first) and phi (second)
  auto basis     = Ikarus::makeBasis(gridView, composite(lagrange(polynomialOrderW), lagrange(polynomialOrderPhi)));
  auto localView = basis.flat().localView();

  /// global stiffness matrix and force vector
  auto numDofs               = basis.flat().size();
  Eigen::VectorXd FExtGlobal = Eigen::VectorXd::Zero(numDofs);
  Eigen::MatrixXd KGlobal    = Eigen::MatrixXd::Zero(numDofs, numDofs);
  Eigen::MatrixXd KLocal;

  for (auto &ele : elements(gridView)) {
    localView.bind(ele);

    /// Define the integration rule
    const auto &rule
        = Dune::QuadratureRules<double, 1>::rule(ele.type(), maxOrderIntegration, Dune::QuadratureType::GaussLegendre);

    /// get local stiffness matrix
    TimoshenkoBeamStiffness(KLocal, localView, ele, rule, C);

    /// Adding local stiffness the global stiffness
    for (auto i = 0U; i < localView.size(); ++i)
      for (auto j = 0U; j < localView.size(); ++j)
        KGlobal(localView.index(i)[0], localView.index(j)[0]) += KLocal(i, j);
  }

  /// apply load on the right-hand side
  FExtGlobal(getGlobalDofId(TimoshenkoBeam::w, basis, L)) = F;

  /// clamp left-hand side
  std::vector<unsigned int> fixedDofs{getGlobalDofId(TimoshenkoBeam::w, basis, 0.0),
                                      getGlobalDofId(TimoshenkoBeam::phi, basis, 0.0)};
  for (auto dof : fixedDofs) {
    KGlobal.col(dof).setZero();
    KGlobal.row(dof).setZero();
    KGlobal(dof, dof) = 1.0;
  }

  /// solve the linear system
  auto linSolver = Ikarus::ILinearSolver<double>(Ikarus::SolverTypeTag::d_LDLT);
  linSolver.factorize(KGlobal);
  Eigen::VectorXd dGlobal;
  linSolver.solve(dGlobal, FExtGlobal);
  /// analytical solution

  // std::cout << "Bernoulli solution for displacement at L: " << F * L * L * L
  // / (3.0 * EI) << "\n";

  // plot the result
  plotDeformedTimoschenkoBeam(gridView, basis, dGlobal, EI, GA, L, F);
}

int main(int argc, char **argv) {
  Ikarus::init(argc, argv);
  exampleTimoshenkoBeam(1, 1, 10);
  exampleTimoshenkoBeam(2, 1, 10);
  exampleTimoshenkoBeam(2, 2, 10);
  exampleTimoshenkoBeam(3, 2, 10);
}
