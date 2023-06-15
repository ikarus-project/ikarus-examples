// SPDX-FileCopyrightText: 2022 The Ikarus Developers mueller@ibb.uni-stuttgart.de
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <config.h>

#include <dune/alugrid/grid.hh>
#include <dune/fufem/boundarypatch.hh>
#include <dune/fufem/dunepython.hh>
#include <dune/fufem/functiontools/boundarydofs.hh>
#include <dune/functions/functionspacebases/basistags.hh>
#include <dune/functions/functionspacebases/boundarydofs.hh>
#include <dune/functions/functionspacebases/lagrangebasis.hh>
#include <dune/functions/functionspacebases/powerbasis.hh>
#include <dune/grid/yaspgrid.hh>
#include <dune/iga/nurbsbasis.hh>
#include <dune/iga/nurbsgrid.hh>

#include "spdlog/spdlog.h"

#include <Eigen/Core>

#include <ikarus/assembler/simpleAssemblers.hh>
#include <ikarus/controlRoutines/loadControl.hh>
#include <ikarus/finiteElements/feRequirements.hh>
#include <ikarus/finiteElements/mechanics/nonLinearElastic.hh>
#include <ikarus/linearAlgebra/dirichletValues.hh>
#include <ikarus/linearAlgebra/nonLinearOperator.hh>
#include <ikarus/solver/nonLinearSolver/newtonRaphson.hh>
#include <ikarus/solver/nonLinearSolver/trustRegion.hh>
#include <ikarus/utils/algorithms.hh>
#include <ikarus/utils/basis.hh>
#include <ikarus/utils/drawing/griddrawer.hh>
#include <ikarus/utils/duneUtilities.hh>
#include <ikarus/utils/init.hh>
#include <ikarus/utils/observer/controlVTKWriter.hh>
#include <ikarus/utils/observer/nonLinearSolverLogger.hh>

// The following grid types (gridType) are included in this example
// 0 - ALUGrid
// 1 - YaspGrid
// 2 - NURBSGrid

// The following solver types (solverType) are included in this example
// 0 - Newton Raphson method
// 1 - Trust region method

#define gridType 2
#define solverType 0

int main(int argc, char **argv) {
  Ikarus::init(argc, argv);
  using namespace Ikarus;
  constexpr int gridDim = 2;

#if gridType == 0
  using Grid = Dune::ALUGrid<gridDim, 2, Dune::simplex, Dune::conforming>;
  auto grid  = Dune::GmshReader<Grid>::read("auxiliaryFiles/unstructuredTrianglesfine.msh", false);
  grid->globalRefine(1);
#endif

#if gridType == 1
  using Grid        = Dune::YaspGrid<gridDim>;
  const double L    = 1;
  const double h    = 1;
  const size_t elex = 10;
  const size_t eley = 10;

  Dune::FieldVector<double, 2> bbox       = {L, h};
  std::array<int, 2> elementsPerDirection = {elex, eley};
  auto grid                               = std::make_shared<Grid>(bbox, elementsPerDirection);
#endif

#if gridType == 2
  constexpr auto dimworld              = 2;
  const std::array<int, gridDim> order = {2, 2};

  const std::array<std::vector<double>, gridDim> knotSpans = {{{0, 0, 0, 1, 1, 1}, {0, 0, 0, 1, 1, 1}}};

  using ControlPoint = Dune::IGA::NURBSPatchData<gridDim, dimworld>::ControlPointType;

  const std::vector<std::vector<ControlPoint>> controlPoints
      = {{{.p = {0, 0}, .w = 5}, {.p = {0.5, 0}, .w = 1}, {.p = {1, 0}, .w = 1}},
         {{.p = {0, 0.5}, .w = 1}, {.p = {0.5, 0.5}, .w = 10}, {.p = {1, 0.5}, .w = 1}},
         {{.p = {0, 1}, .w = 1}, {.p = {0.5, 1}, .w = 1}, {.p = {1, 1}, .w = 1}}};

  std::array<int, gridDim> dimsize = {(int)(controlPoints.size()), (int)(controlPoints[0].size())};

  auto controlNet = Dune::IGA::NURBSPatchData<gridDim, dimworld>::ControlPointNetType(dimsize, controlPoints);
  using Grid      = Dune::IGA::NURBSGrid<gridDim, dimworld>;

  Dune::IGA::NURBSPatchData<gridDim, dimworld> patchData;
  patchData.knotSpans     = knotSpans;
  patchData.degree        = order;
  patchData.controlPoints = controlNet;
  auto grid               = std::make_shared<Grid>(patchData);
  grid->globalRefine(2);
#endif

  auto gridView        = grid->leafGridView();
  const auto &indexSet = gridView.indexSet();

  Dune::BitSetVector<1> neumannVertices(gridView.size(2), false);

  std::string lambdaNeumannVertices = std::string("lambda x: ( x[0]>0.999 )");
  Python::start();
  Python::Reference main = Python::import("__main__");
  Python::run("import math");

  Python::runStream() << std::endl << "import sys" << std::endl << "import os" << std::endl;

  auto pythonNeumannVertices = Python::make_function<bool>(Python::evaluate(lambdaNeumannVertices));

  for (auto &&vertex : vertices(gridView)) {
    bool isNeumann                          = pythonNeumannVertices(vertex.geometry().corner(0));
    neumannVertices[indexSet.index(vertex)] = isNeumann;
  }

  BoundaryPatch<decltype(gridView)> neumannBoundary(gridView, neumannVertices);

  using namespace Dune::Functions::BasisFactory;

#if gridType == 0 or gridType == 1
  auto basis = Ikarus::makeBasis(gridView, power<gridDim>(lagrange<1>()));
#endif
#if (gridType == 2)
  auto basis = Ikarus::makeBasis(gridView, power<gridDim>(nurbs()));
#endif

  std::cout << "This gridview contains: " << std::endl;
  std::cout << gridView.size(2) << " vertices" << std::endl;
  std::cout << gridView.size(1) << " edges" << std::endl;
  std::cout << gridView.size(0) << " elements" << std::endl;
  std::cout << basis.flat().size() << " Dofs" << std::endl;

  //  draw(gridView);

  auto localView = basis.flat().localView();

  auto matParameter = Ikarus::toLamesFirstParameterAndShearModulus({.emodul = 1000, .nu = 0.3});

  Ikarus::StVenantKirchhoff matSVK(matParameter);
  auto reducedMat = planeStress(matSVK, 1e-8);
  std::vector<Ikarus::NonLinearElastic<decltype(basis), decltype(reducedMat)>> fes;

  auto volumeLoad = [](auto &globalCoord, auto &lamb) {
    Eigen::Vector2d fext;
    fext.setZero();
    return fext;
  };

  auto neumannBoundaryLoad = [](auto &globalCoord, auto &lamb) {
    Eigen::Vector2d fext;
    fext.setZero();
    fext[1] = lamb / 40;
    return fext;
  };

  for (auto &element : elements(gridView))
    fes.emplace_back(basis, element, reducedMat, volumeLoad, &neumannBoundary, neumannBoundaryLoad);

  auto basisP = std::make_shared<const decltype(basis)>(basis);
  Ikarus::DirichletValues dirichletValues(basisP->flat());

  dirichletValues.fixBoundaryDOFs([&](auto &dirichletFlags, auto &&localIndex, auto &&localView, auto &&intersection) {
    if (std::abs(intersection.geometry().center()[1]) < 1e-8) dirichletFlags[localView.index(localIndex)] = true;
  });

  auto sparseAssembler = SparseFlatAssembler(fes, dirichletValues);

  Eigen::VectorXd d;
  d.setZero(basis.flat().size());
  double lambda = 0.0;

  auto req = FErequirements().addAffordance(Ikarus::AffordanceCollections::elastoStatics);

  auto residualFunction = [&](auto &&disp, auto &&lambdaLocal) -> auto & {
    req.insertGlobalSolution(Ikarus::FESolutions::displacement, disp)
        .insertParameter(Ikarus::FEParameter::loadfactor, lambdaLocal);
    return sparseAssembler.getVector(req);
  };

  auto KFunction = [&](auto &&disp, auto &&lambdaLocal) -> auto & {
    req.insertGlobalSolution(Ikarus::FESolutions::displacement, disp)
        .insertParameter(Ikarus::FEParameter::loadfactor, lambdaLocal);
    return sparseAssembler.getMatrix(req);
  };

  auto energyFunction = [&](auto &&disp, auto &&lambdaLocal) -> auto & {
    req.insertGlobalSolution(Ikarus::FESolutions::displacement, disp)
        .insertParameter(Ikarus::FEParameter::loadfactor, lambdaLocal);
    return sparseAssembler.getScalar(req);
  };

  auto nonLinOp
      = Ikarus::NonLinearOperator(functions(energyFunction, residualFunction, KFunction), parameter(d, lambda));

  auto linSolver = Ikarus::ILinearSolver<double>(Ikarus::SolverTypeTag::sd_UmfPackLU);

#if solverType == 0
  auto nr = Ikarus::makeNewtonRaphson(nonLinOp.subOperator<1, 2>(), std::move(linSolver));
#endif
#if solverType == 1
  auto nr = Ikarus::makeTrustRegion(nonLinOp);
  nr->setup({.verbosity = 1,
             .maxiter   = 30,
             .grad_tol  = 1e-8,
             .corr_tol  = 1e-8,
             .useRand   = false,
             .rho_reg   = 1e6,
             .Delta0    = 1});
#endif

  auto nonLinearSolverObserver = std::make_shared<NonLinearSolverLogger>();

  auto vtkWriter = std::make_shared<ControlSubsamplingVertexVTKWriter<std::remove_cvref_t<decltype(basis.flat())>>>(
      basis.flat(), d, 2);
  vtkWriter->setFileNamePrefix("iks006_nonlinear2DSolid");
  vtkWriter->setFieldInfo("Displacement", Dune::VTK::FieldInfo::Type::vector, 2);
  nr->subscribeAll(nonLinearSolverObserver);

  auto lc = Ikarus::LoadControl(nr, 20, {0, 2000});

  lc.subscribeAll(vtkWriter);
  std::cout << "Energy before: " << nonLinOp.value() << std::endl;
  lc.run();
  nonLinOp.update<0>();
  std::cout << "Energy after: " << nonLinOp.value() << std::endl;
}
