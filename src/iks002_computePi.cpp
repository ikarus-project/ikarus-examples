// SPDX-FileCopyrightText: 2022 The Ikarus Developers mueller@ibb.uni-stuttgart.de
// SPDX-License-Identifier: LGPL-2.1-or-later

#include <numbers>

#include <dune/alugrid/grid.hh>
#include <dune/geometry/quadraturerules.hh>
#include <dune/grid/io/file/vtk/vtkwriter.hh>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <ikarus/utils/drawing/griddrawer.hh>

int main() {
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
    std::cout << area << " " << std::numbers::pi << std::endl;
  }
  /// write element areas to vtk
  std::vector<double> areas;
  areas.resize(gridView.size(0));

  auto &indexSet = gridView.indexSet();
  for (auto &ele : elements(gridView))
    areas[indexSet.index(ele)] = ele.geometry().volume();

  Dune::VTKWriter vtkWriter(gridView);
  vtkWriter.addCellData(areas, "area", 1);
  vtkWriter.write("iks002_computePi");

  /// Calculate circumference and compare to pi
  double circumference = 0.0;
  for (auto &element : elements(gridView))
    if (element.hasBoundaryIntersections())
      for (auto &intersection : intersections(gridView, element))
        if (intersection.boundary()) circumference += intersection.geometry().volume();

  std::cout << circumference << " " << std::numbers::pi << std::endl;
}
