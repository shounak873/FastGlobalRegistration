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
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <cstdlib>
using namespace std;


int main(int argc, char *argv[])
{
	if (argc != 4)
	{
        printf("Usage ::\n");
		printf("%s [feature_01] [feature_02] [transform_output_txt]\n", argv[0]);
		return 0;
	}

	string fname = "/home/navlab-shounak/FastGlobalRegistration/source/FastGlobalRegistration/table.txt";

	vector<vector<double>> content;
	vector<double> row;
	string line, word;
	fstream file (fname, ios::in);
	if(file.is_open())
	{
		while(getline(file, line))
		{
			row.clear();
			stringstream str(line);
			while(getline(str, word, ' '))
				row.push_back(stof(word));
			content.push_back(row);
		}
	}
	else{
		cout<<"Could not open the file\n";
	}
	std::cout << " Finished reading " << std::endl;

	fgr::CApp app;
	app.ReadFeature(argv[1]);
	app.ReadFeature(argv[2]);
	app.NormalizePoints();
	app.AdvancedMatching();
	app.OptimizePairwise(content);
	app.WriteTrans(argv[3]);


	// vector<double> resvec{0.0001, 0.02, 0.3, 0.05};
	// for(auto it : resvec){
	// 	std::cout << app.robustcost(it, 1.0, 1.75) << std::endl;
	// }

	return 0;
}
