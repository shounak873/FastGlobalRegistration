// ----------------------------------------------------------------------------
// -                       Fast Global Registration                           -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) Intel Corporation 2016
// Qianyi Zhou <Qianyi.Zhou@gmail.com>
// Jaesik Park <syncle@gmail.com>
// Vladlen Koltun <vkoltun@gmail.com>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------
#include <stdio.h>
#include "app.h"


int main(int argc, char *argv[])
{
	if (argc != 4)
	{
        printf("Usage ::\n");
		printf("%s [feature_01] [feature_02] [transform_output_txt]\n", argv[0]);
		return 0;
	}
	fgr::CApp app;
	// std::ifstream file2("table.txt");
	// //---------------------------------------------------------------------
	// // read the constant values from txt file
	// std::vector<std::vector<double> > constTable;
	// std::string line;
	// double value2;
	// int rowNum = 0;
	// // read table into matrix
	// while(std::getline(file2, line)) {
	// 		std::vector<double> row;
	// 		std::cout << "row size" << row.size() << std::endl;
	// 		std::istringstream iss(line);
	// 		while(iss >> value2){
	// 				row.push_back(value2);
	// 		}
	// 		constTable.push_back(row);
	// 		rowNum = rowNum + 1;
	// }
	// std::cout << "Row numbers - " << rowNum << std::endl;
	// std::cout << "Col numbers - " << constTable[0].size() << std::endl;



	app.ReadFeature(argv[1]);
	app.ReadFeature(argv[2]);
	app.NormalizePoints();
	app.AdvancedMatching();
	app.OptimizePairwise();
	app.WriteTrans(argv[3]);

	return 0;
}
