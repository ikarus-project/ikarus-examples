// SPDX-FileCopyrightText: 2021-2025 The Ikarus Developers ikarus@ibb.uni-stuttgart.de
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "config.h"

#include <ikarus/assembler/simpleassemblers.hh>
#include <ikarus/solver/nonlinearsolver/newtonraphson.hh>
#include <ikarus/utils/broadcaster/broadcastermessages.hh>
#include <ikarus/utils/differentiablefunction.hh>
#include <ikarus/utils/init.hh>
#include <ikarus/utils/listener/listener.hh>

auto func(const double& x) { return 0.5 * x * x + x - 2; }
auto funcDerivative(const double& x) { return x + 1; }

void newtonRaphsonVeryBasicExample() {
  double x               = 13;
  const double eps       = 1e-10;
  const int maxIter      = 20;
  const double xExpected = std::sqrt(5.0) - 1.0;

  auto fvLambda  = [&](auto&& x) { return func(x); };
  auto dfvLambda = [&](auto&& x) { return funcDerivative(x); };
  auto nonLinOp  = Ikarus::makeDifferentiableFunction(Ikarus::functions(fvLambda, dfvLambda), x);

  /// Standard implementation
  int iterCount = 1;
  while (abs(nonLinOp(x)) > eps and iterCount <= maxIter) {
    const auto f  = nonLinOp(x);
    const auto df = derivative(nonLinOp)(x);
    x -= f / df;
    iterCount++;

    std::cout << "f, value: " << f << "\n";
    std::cout << "nonlinearOperator, x: " << x << "\n";
  }

  /// Implementation with Ikarus
  Ikarus::NewtonRaphson nr(nonLinOp);
  nr.setup({eps, maxIter});
  const auto solverInfo = nr.solve(x);

  std::cout << "success: " << solverInfo.success << "\n";
  std::cout << "iterations: " << solverInfo.iterations << "\n";
  std::cout << "residuum: " << solverInfo.residualNorm << "\n";
  std::cout << "solution: " << x << "\n";
  std::cout << "expected solution: " << xExpected << "\n";
}

class OurFirstListener : public Ikarus::Listener
{
public:
  template <typename BC>
  OurFirstListener& subscribeTo(BC& bc) {
    this->subscribe(bc, [&](Ikarus::NonLinearSolverMessages message) { this->updateImpl(message); });
    return *this;
  }

  void updateImpl(Ikarus::NonLinearSolverMessages message) {
    if (message == Ikarus::NonLinearSolverMessages::ITERATION_STARTED)
      std::cout << "Iteration started.\n";
  }
};

void newtonRaphsonBasicExampleWithLogger() {
  double x = 13;

  auto fvLambda  = [&](auto&& x) { return func(x); };
  auto dfvLambda = [&](auto&& x) { return funcDerivative(x); };
  auto nonLinOp  = Ikarus::makeDifferentiableFunction(Ikarus::functions(fvLambda, dfvLambda), x);

  const double eps       = 1e-10;
  const int maxIter      = 20;
  const double xExpected = std::sqrt(5.0) - 1.0;

  Ikarus::NewtonRaphson nr(nonLinOp);
  nr.setup({eps, maxIter});

  // create observer and subscribe to Newton-Rhapson
  auto ourSimpleListener = OurFirstListener();
  ourSimpleListener.subscribeTo(nr);
  // nr.subscribeAll(ourSimpleObserver);
  // auto nonLinearSolverObserver = std::make_shared<NonLinearSolverLogger>();
  // nr.subscribe(Ikarus::NonLinearSolverMessages::FINISHED_SUCESSFULLY,
  // nonLinearSolverObserver); nr.subscribeAll(nonLinearSolverObserver);

  const auto solverInfo = nr.solve(x);
  if (solverInfo.success)
    std::cout << "solution: " << x << "\n";
  else
    std::cout << "The Newton-Raphson procedure failed to converge" << std::endl;
}

int main(int argc, char** argv) {
  Ikarus::init(argc, argv);
  newtonRaphsonVeryBasicExample();
  std::cout << "\nWith Logger\n\n";
  newtonRaphsonBasicExampleWithLogger();
}
