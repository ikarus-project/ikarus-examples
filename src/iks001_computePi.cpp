// SPDX-FileCopyrightText: 2022 The Ikarus Developers mueller@ibb.uni-stuttgart.de
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <config.h>

#include <numbers>

#include <dune/alugrid/grid.hh>
#include <dune/geometry/quadraturerules.hh>
#include <dune/grid/common/boundarysegment.hh>
#include <dune/grid/io/file/vtk/vtkwriter.hh>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <ikarus/utils/drawing/griddrawer.hh>
#include <ikarus/utils/init.hh>

void boundaryUnawareRefinedCircle() {
  std::cout << std::endl << "Comparing the values of Pi with global refinements of unit circle" << std::endl;

  constexpr int gridDim = 2;  // (1)
  using Grid            = Dune::ALUGrid<gridDim, 2, Dune::simplex, Dune::conforming>;
  auto grid             = Dune::GmshReader<Grid>::read("auxiliaryFiles/circleCoarse.msh", false);
  auto gridView         = grid->leafGridView();  // (2)

  // draw(gridView);

  /// Calculate area from volume function of elements
  double area = 0.0;

  /// Naive refinement of grid and compare calculated area to pi
  for (int i = 0; i < 3; ++i) {
    area = 0.0;
    grid->globalRefine(1);

    auto gridViewRefined = grid->leafGridView();
    std::cout << "This gridview contains: ";
    std::cout << gridViewRefined.size(0) << " elements" << std::endl;
    //    draw(gridViewRefined);
    for (auto &element : elements(gridViewRefined)) {
      area += element.geometry().volume();
    }
    std::cout << std::setprecision(10) << "Area: " << area << " Pi: " << std::numbers::pi << std::endl;
  }
  /// write element areas to vtk
  std::vector<double> areas;
  areas.resize(gridView.size(0));

  auto &indexSet = gridView.indexSet();
  for (auto &ele : elements(gridView))
    areas[indexSet.index(ele)] = ele.geometry().volume();

  Dune::VTKWriter vtkWriter(gridView);
  vtkWriter.addCellData(areas, "area", 1);
  vtkWriter.write("iks001_computePi");

  /// Calculate circumference and compare to pi
  double circumference = 0.0;
  for (auto &element : elements(gridView))
    if (element.hasBoundaryIntersections())
      for (auto &intersection : intersections(gridView, element))
        if (intersection.boundary()) circumference += intersection.geometry().volume();

  std::cout << std::setprecision(10) << "Circumference/2: " << circumference / 2 << " Pi: " << std::numbers::pi
            << std::endl;
}

struct UnitCircleBoundary : Dune::BoundarySegment<2, 2, double> {
  UnitCircleBoundary(const Dune::FieldVector<double, 2> &a, const Dune::FieldVector<double, 2> &b) : corners{{a, b}} {}
  Dune::FieldVector<double, 2> operator()(const Dune::FieldVector<double, 1> &local) const override {
    Dune::FieldVector<double, 2> result = {0, 0};
    double omega                        = std::acos(corners[0] * corners[1]);
    return std::sin((1 - local[0]) * omega) / sin(omega) * corners[0] + sin(local[0] * omega) / sin(omega) * corners[1];
  }

  std::array<Dune::FieldVector<double, 2>, 2> corners;
};

void boundaryAwareRefinedCircle() {
  std::cout << std::endl
            << "Comparing the values of Pi with local refinement on the boundary of unit circle" << std::endl;
  /// Create a grid from 6 triangles align in unit disc
  using namespace Dune;
  constexpr int gridDim = 2;
  Dune::GridFactory<Dune::ALUGrid<gridDim, 2, Dune::simplex, Dune::conforming>> gridFactory;
  Eigen::Vector2d v(1, 0);
  std::array<FieldVector<double, 2>, 6> corners0;
  Eigen::Rotation2D<double> R;
  R.angle() = 0.0;
  for (auto &corner : corners0) {
    Eigen::Vector2d a = R * v;
    corner[0]         = a[0];
    corner[1]         = a[1];
    R.angle() += 60.0 / 180.0 * std::numbers::pi;
  }

  gridFactory.insertVertex({0, 0});
  gridFactory.insertVertex(corners0[0]);
  gridFactory.insertVertex(corners0[1]);
  gridFactory.insertVertex(corners0[2]);
  gridFactory.insertVertex(corners0[3]);
  gridFactory.insertVertex(corners0[4]);
  gridFactory.insertVertex(corners0[5]);

  gridFactory.insertElement(Dune::GeometryTypes::triangle, {0, 1, 2});
  gridFactory.insertElement(Dune::GeometryTypes::triangle, {0, 2, 3});
  gridFactory.insertElement(Dune::GeometryTypes::triangle, {0, 3, 4});
  gridFactory.insertElement(Dune::GeometryTypes::triangle, {0, 4, 5});
  gridFactory.insertElement(Dune::GeometryTypes::triangle, {0, 5, 6});
  gridFactory.insertElement(Dune::GeometryTypes::triangle, {0, 6, 1});

  /// Create boundary segments which map the boundaries onto the unit circle
  gridFactory.insertBoundarySegment({1, 2}, std::make_shared<UnitCircleBoundary>(corners0[0], corners0[1]));
  gridFactory.insertBoundarySegment({2, 3}, std::make_shared<UnitCircleBoundary>(corners0[1], corners0[2]));
  gridFactory.insertBoundarySegment({3, 4}, std::make_shared<UnitCircleBoundary>(corners0[2], corners0[3]));
  gridFactory.insertBoundarySegment({4, 5}, std::make_shared<UnitCircleBoundary>(corners0[3], corners0[4]));
  gridFactory.insertBoundarySegment({5, 6}, std::make_shared<UnitCircleBoundary>(corners0[4], corners0[5]));
  gridFactory.insertBoundarySegment({6, 1}, std::make_shared<UnitCircleBoundary>(corners0[5], corners0[0]));

  auto grid     = gridFactory.createGrid();
  auto gridView = grid->leafGridView();
  // draw(gridView);

  double area = 0.0;
  for (int i = 0; i < 10; ++i) {
    area = 0.0;
    /// Refine grid entities if they live at the boundary
    //    grid->globalRefine(1);
    for (const auto &ele : elements(grid->leafGridView())) {
      if (ele.hasBoundaryIntersections()) grid->mark(1, ele);
    }
    grid->preAdapt();
    grid->adapt();
    grid->postAdapt();
    auto gridViewRefined = grid->leafGridView();

    std::cout << "This gridview contains: ";
    std::cout << gridViewRefined.size(0) << " elements" << std::endl;

    for (auto &element : elements(gridViewRefined))
      area += element.geometry().volume();

    std::cout << std::setprecision(10) << "Area: " << area << " Pi: " << std::numbers::pi << std::endl;
    // draw(gridViewRefined);
  }
  /// Calculate circumference and compare to pi
  double circumference = 0.0;
  for (auto &element : elements(grid->leafGridView()))
    if (element.hasBoundaryIntersections())
      for (auto &intersection : intersections(grid->leafGridView(), element))
        if (intersection.boundary()) circumference += intersection.geometry().volume();

  std::cout << std::setprecision(10) << "Circumference/2: " << circumference / 2 << " Pi: " << std::numbers::pi
            << std::endl;
}

int main(int argc, char **argv) {
  Ikarus::init(argc, argv);
  boundaryUnawareRefinedCircle();
  boundaryAwareRefinedCircle();
}
