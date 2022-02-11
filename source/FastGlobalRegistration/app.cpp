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

#include "app.h"
#include <math.h>

using namespace Eigen;
using namespace std;
using namespace fgr;

void CApp::ReadFeature(const char* filepath)
{
	Points pts;
	Feature feat;
	ReadFeature(filepath, pts, feat);
	LoadFeature(pts,feat);
}

void CApp::LoadFeature(const Points& pts, const Feature& feat)
{
	pointcloud_.push_back(pts);
	features_.push_back(feat);
}

void CApp::ReadFeature(const char* filepath, Points& pts, Feature& feat)
{
	printf("ReadFeature ... ");
	FILE* fid = fopen(filepath, "rb");
	int nvertex;
	fread(&nvertex, sizeof(int), 1, fid);
	int ndim;
	fread(&ndim, sizeof(int), 1, fid);

	// read from feature file and fill out pts and feat
	for (int v = 0; v < nvertex; v++)	{

		Vector3f pts_v;
		fread(&pts_v(0), sizeof(float), 3, fid);

		VectorXf feat_v(ndim);
		fread(&feat_v(0), sizeof(float), ndim, fid);

		pts.push_back(pts_v);
		feat.push_back(feat_v);
	}
	fclose(fid);
	printf("%d points with %d feature dimensions.\n", nvertex, ndim);
}

template <typename T>
void CApp::BuildKDTree(const vector<T>& data, KDTree* tree)
{
	int rows, dim;
	rows = (int)data.size();
	dim = (int)data[0].size();
	std::vector<float> dataset(rows * dim);
	flann::Matrix<float> dataset_mat(&dataset[0], rows, dim);
	for (int i = 0; i < rows; i++)
		for (int j = 0; j < dim; j++)
			dataset[i * dim + j] = data[i][j];
	KDTree temp_tree(dataset_mat, flann::KDTreeSingleIndexParams(15));
	temp_tree.buildIndex();
	*tree = temp_tree;
}

template <typename T>
void CApp::SearchKDTree(KDTree* tree, const T& input,
							std::vector<int>& indices,
							std::vector<float>& dists, int nn)
{
	int rows_t = 1;
	int dim = input.size();

	std::vector<float> query;
	query.resize(rows_t*dim);
	for (int i = 0; i < dim; i++)
		query[i] = input(i);
	flann::Matrix<float> query_mat(&query[0], rows_t, dim);

	indices.resize(rows_t*nn);
	dists.resize(rows_t*nn);
	flann::Matrix<int> indices_mat(&indices[0], rows_t, nn);
	flann::Matrix<float> dists_mat(&dists[0], rows_t, nn);

	tree->knnSearch(query_mat, indices_mat, dists_mat, nn, flann::SearchParams(128));
}

void CApp::AdvancedMatching()
{
	int fi = 0;
	int fj = 1;

	printf("Advanced matching : [%d - %d]\n", fi, fj);
	bool swapped = false;

	if (pointcloud_[fj].size() > pointcloud_[fi].size())
	{
		int temp = fi;
		fi = fj;
		fj = temp;
		swapped = true;
	}

	int nPti = pointcloud_[fi].size();
	int nPtj = pointcloud_[fj].size();

	///////////////////////////
	/// BUILD FLANNTREE
	///////////////////////////

	KDTree feature_tree_i(flann::KDTreeSingleIndexParams(15));
	BuildKDTree(features_[fi], &feature_tree_i);

	KDTree feature_tree_j(flann::KDTreeSingleIndexParams(15));
	BuildKDTree(features_[fj], &feature_tree_j);

	bool crosscheck = true;
	bool tuple = true;

	std::vector<int> corres_K, corres_K2;
	std::vector<float> dis;
	std::vector<int> ind;

	std::vector<std::pair<int, int> > corres;
	std::vector<std::pair<int, int> > corres_cross;
	std::vector<std::pair<int, int> > corres_ij;
	std::vector<std::pair<int, int> > corres_ji;

	///////////////////////////
	/// INITIAL MATCHING
	///////////////////////////

	std::vector<int> i_to_j(nPti, -1);
	for (int j = 0; j < nPtj; j++)
	{
		SearchKDTree(&feature_tree_i, features_[fj][j], corres_K, dis, 1);
		int i = corres_K[0];
		if (i_to_j[i] == -1)
		{
			SearchKDTree(&feature_tree_j, features_[fi][i], corres_K, dis, 1);
			int ij = corres_K[0];
			i_to_j[i] = ij;
		}
		corres_ji.push_back(std::pair<int, int>(i, j));
	}

	for (int i = 0; i < nPti; i++)
	{
		if (i_to_j[i] != -1)
			corres_ij.push_back(std::pair<int, int>(i, i_to_j[i]));
	}

	int ncorres_ij = corres_ij.size();
	int ncorres_ji = corres_ji.size();

	// corres = corres_ij + corres_ji;
	for (int i = 0; i < ncorres_ij; ++i)
		corres.push_back(std::pair<int, int>(corres_ij[i].first, corres_ij[i].second));
	for (int j = 0; j < ncorres_ji; ++j)
		corres.push_back(std::pair<int, int>(corres_ji[j].first, corres_ji[j].second));

	printf("Number of points that remain: %d\n", (int)corres.size());

	///////////////////////////
	/// CROSS CHECK
	/// input : corres_ij, corres_ji
	/// output : corres
	///////////////////////////
	if (crosscheck)
	{
		printf("\t[cross check] ");

		// build data structure for cross check
		corres.clear();
		corres_cross.clear();
		std::vector<std::vector<int> > Mi(nPti);
		std::vector<std::vector<int> > Mj(nPtj);

		int ci, cj;
		for (int i = 0; i < ncorres_ij; ++i)
		{
			ci = corres_ij[i].first;
			cj = corres_ij[i].second;
			Mi[ci].push_back(cj);
		}
		for (int j = 0; j < ncorres_ji; ++j)
		{
			ci = corres_ji[j].first;
			cj = corres_ji[j].second;
			Mj[cj].push_back(ci);
		}

		// cross check
		for (int i = 0; i < nPti; ++i)
		{
			for (int ii = 0; ii < Mi[i].size(); ++ii)
			{
				int j = Mi[i][ii];
				for (int jj = 0; jj < Mj[j].size(); ++jj)
				{
					if (Mj[j][jj] == i)
					{
						corres.push_back(std::pair<int, int>(i, j));
						corres_cross.push_back(std::pair<int, int>(i, j));
					}
				}
			}
		}
		printf("Number of points that remain after cross-check: %d\n", (int)corres.size());
	}

	///////////////////////////
	/// TUPLE CONSTRAINT
	/// input : corres
	/// output : corres
	///////////////////////////
	if (tuple)
	{
		srand(time(NULL));

		printf("\t[tuple constraint] ");
		int rand0, rand1, rand2;
		int idi0, idi1, idi2;
		int idj0, idj1, idj2;
		float scale = tuple_scale_;
		int ncorr = corres.size();
		int number_of_trial = ncorr * 100;
		std::vector<std::pair<int, int> > corres_tuple;

		int cnt = 0;
		int i;
		for (i = 0; i < number_of_trial; i++)
		{
			rand0 = rand() % ncorr;
			rand1 = rand() % ncorr;
			rand2 = rand() % ncorr;

			idi0 = corres[rand0].first;
			idj0 = corres[rand0].second;
			idi1 = corres[rand1].first;
			idj1 = corres[rand1].second;
			idi2 = corres[rand2].first;
			idj2 = corres[rand2].second;

			// collect 3 points from i-th fragment
			Eigen::Vector3f pti0 = pointcloud_[fi][idi0];
			Eigen::Vector3f pti1 = pointcloud_[fi][idi1];
			Eigen::Vector3f pti2 = pointcloud_[fi][idi2];

			float li0 = (pti0 - pti1).norm();
			float li1 = (pti1 - pti2).norm();
			float li2 = (pti2 - pti0).norm();

			// collect 3 points from j-th fragment
			Eigen::Vector3f ptj0 = pointcloud_[fj][idj0];
			Eigen::Vector3f ptj1 = pointcloud_[fj][idj1];
			Eigen::Vector3f ptj2 = pointcloud_[fj][idj2];

			float lj0 = (ptj0 - ptj1).norm();
			float lj1 = (ptj1 - ptj2).norm();
			float lj2 = (ptj2 - ptj0).norm();

			if ((li0 * scale < lj0) && (lj0 < li0 / scale) &&
				(li1 * scale < lj1) && (lj1 < li1 / scale) &&
				(li2 * scale < lj2) && (lj2 < li2 / scale))
			{
				corres_tuple.push_back(std::pair<int, int>(idi0, idj0));
				corres_tuple.push_back(std::pair<int, int>(idi1, idj1));
				corres_tuple.push_back(std::pair<int, int>(idi2, idj2));
				cnt++;
			}

			if (cnt >= tuple_max_cnt_)
				break;
		}

		printf("%d tuples (%d trial, %d actual).\n", cnt, number_of_trial, i);
		corres.clear();

		for (int i = 0; i < corres_tuple.size(); ++i)
			corres.push_back(std::pair<int, int>(corres_tuple[i].first, corres_tuple[i].second));
	}

	if (swapped)
	{
		std::vector<std::pair<int, int> > temp;
		for (int i = 0; i < corres.size(); i++)
			temp.push_back(std::pair<int, int>(corres[i].second, corres[i].first));
		corres.clear();
		corres = temp;
	}

	printf("\t[final] matches %d.\n", (int)corres.size());
	corres_ = corres;
}

// Normalize scale of points.
// X' = (X-\mu)/scale
void CApp::NormalizePoints()
{
	int num = 2;
	float scale = 0;

	Means.clear();

	for (int i = 0; i < num; ++i)
	{
		float max_scale = 0;

		// compute mean
		Vector3f mean;
		mean.setZero();

		int npti = pointcloud_[i].size();
		for (int ii = 0; ii < npti; ++ii)
		{
			Eigen::Vector3f p(pointcloud_[i][ii](0), pointcloud_[i][ii](1), pointcloud_[i][ii](2));
			mean = mean + p;
		}
		mean = mean / npti;
		Means.push_back(mean);

		printf("normalize points :: mean[%d] = [%f %f %f]\n", i, mean(0), mean(1), mean(2));

		for (int ii = 0; ii < npti; ++ii)
		{
			pointcloud_[i][ii](0) -= mean(0);
			pointcloud_[i][ii](1) -= mean(1);
			pointcloud_[i][ii](2) -= mean(2);
		}

		// compute scale
		for (int ii = 0; ii < npti; ++ii)
		{
			Eigen::Vector3f p(pointcloud_[i][ii](0), pointcloud_[i][ii](1), pointcloud_[i][ii](2));
			float temp = p.norm(); // because we extract mean in the previous stage.
			if (temp > max_scale)
				max_scale = temp;
		}

		if (max_scale > scale)
			scale = max_scale;
	}

	//// mean of the scale variation
	if (use_absolute_scale_) {
		GlobalScale = 1.0f;
		StartScale = scale;
	} else {
		GlobalScale = scale; // second choice: we keep the maximum scale.
		StartScale = 1.0f;
	}
	printf("normalize points :: global scale : %f\n", GlobalScale);

	for (int i = 0; i < num; ++i)
	{
		int npti = pointcloud_[i].size();
		for (int ii = 0; ii < npti; ++ii)
		{
			pointcloud_[i][ii](0) /= GlobalScale;
			pointcloud_[i][ii](1) /= GlobalScale;
			pointcloud_[i][ii](2) /= GlobalScale;
		}
	}
}

void CApp::OptimizePairwise()
{
	printf("Pairwise rigid pose optimization\n");

	int ConvergIter = 10;
	double tol = 1e-7;
	std::vector<std::vector<double> > constTable
	{
		{2.1532, 3.2298, 4.3064, 5.3830, 6.4596, 7.5361, 8.6105, 9.6729, 10.7016, 11.6707, 12.5602},
		{2.5066, 3.7599, 5.0132, 6.2662, 7.5134, 8.7357, 9.902, 10.9835, 11.9629, 12.8347, 13.6022},
		{3.2721, 4.8987, 6.4773, 7.9559, 9.2999, 10.4966, 11.5490, 12.4681, 13.2684, 13.9646, 14.5708},
		{4.0455, 5.7775, 7.3265, 8.7042, 9.9239, 11.0001, 11.9472, 12.7794, 13.5102, 14.1520, 14.7161},
		{4.9674, 6.5253, 7.9154, 9.1594, 10.2714, 11.2632, 12.1456, 12.9287, 13.6227, 14.2369, 14.7804},
		{5.7304, 7.0801, 8.3223, 9.4587, 10.4917, 11.4255, 12.2652, 13.0172, 13.6883, 14.2858, 14.817},
		{6.2859, 7.4804, 8.6113, 9.6677, 10.6432, 11.5356, 12.3454, 13.0758, 13.7313, 14.3176, 14.8406},
		{6.6859, 7.7737, 8.8239, 9.8209, 10.7535, 11.6150, 12.4029, 13.1175, 13.7618, 14.3399, 14.8571},
		{6.9804, 7.9943, 8.9852, 9.9374, 10.8372, 11.6751, 12.4461, 13.1488, 13.7845, 14.3565, 14.8694},
		{7.2037, 8.1648, 9.1113, 10.0288, 10.9028, 11.7221, 12.4798, 13.173, 13.8021, 14.3694, 14.8788},
		{7.3777, 8.2997, 9.2121, 10.1022, 10.9556, 11.7599, 12.5068, 13.1925, 13.8161, 14.3796, 14.8863},
		{7.5167, 8.4088, 9.2943, 10.1624, 10.9989, 11.7909, 12.529, 13.2083, 13.8275, 14.3879, 14.8924},
		{7.63, 8.4986, 9.3626, 10.2127, 11.0352, 11.8167, 12.5474, 13.2215, 13.837, 14.3948, 14.8974},
		{7.7241, 8.5738, 9.4202, 10.2552, 11.0659, 11.8387, 12.5631, 13.2327, 13.8451, 14.4006, 14.9016}
	};
	std::cout << "Row numbers - " << constTable.size() << std::endl;
	std::cout << "Col numbers - " << constTable[0].size() << std::endl;

	//---------------------------------------------------------------------
	std::vector<double> alpha{2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0};
	std::vector<double> c{1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0};

	int maxalphaind = 1;
	int maxcind = 0;
	std::cout << "Before optimization" << std::endl;
	std::cout << "Best alpha -- " << alpha[maxalphaind] << endl;
	std::cout << "Best c -- " << c[maxcind] << endl;
	std::cout << " ------------------------ " << std::endl;

	double totallike;
	std::vector<double> likevec;
	std::vector<double> resnormvec;

	Eigen::Matrix4f trans;
	Eigen::Matrix4f pretrans;
	trans.setIdentity();
	double diff = 1.0;
	TransOutput_ = Eigen::Matrix4f::Identity();

	int lenalpha = alpha.size();
	int lenc = c.size();

	// selecting the best alpha and c by maximizing respective negative log-likelihoods
	int i = 0;
	int j = 1;

	// make another copy of pointcloud_[j].
	Points pcj_copy;
	int npcj = pointcloud_[j].size();
	pcj_copy.resize(npcj);
	for (int cnt = 0; cnt < npcj; cnt++)
		pcj_copy[cnt] = pointcloud_[j][cnt];

	// Main iteration cycle starts
	for (int itr = 0; itr < ConvergIter; itr++){
		if(diff > tol){
			pretrans = trans;
			std::cout << "Iteration number outer  -- " << itr << std::endl;
			resnormvec.clear();
			for (int cr = 0; cr < corres_.size(); cr++) {
				int ii = corres_[cr].first;
				int jj = corres_[cr].second;
				Eigen::Vector3f p, q;
				p = pointcloud_[i][ii];
				q = pcj_copy[jj];
				Eigen::Vector3f rpq = p - q;
				double resnorm = rpq.norm();
				// std::cout << "Residual norm - " << resnorm << endl;
				resnormvec.push_back(resnorm);
			}

		    // firstly, keep c constant and maximize with respect to alpha
			likevec.clear();
			for(int ip =0; ip < lenalpha; ip++){
				totallike = 0.0;
				for(auto it2 : resnormvec){
					totallike += exp(-robustcost(it2,c[maxcind], alpha[ip]))/constTable[ip][maxcind];
				}
				// std::cout << "Likelihood for  alpha = " << alpha[ip] << " and "<< " c = "<< c[maxcind] << " is " << totallike << endl;
				likevec.push_back(totallike);
			}

		    std::vector<double>::iterator result;
			// std::cout << "Like vec size " << likevec.size() << std::endl;
		    result = std::max_element(likevec.begin(), likevec.end());
		    maxalphaind = std::distance(likevec.begin(), result);
			std::cout << "Best alpha -- " << alpha[maxalphaind] << endl;
			// std::cout << " ------------------------ "  << std::endl;

			// secondly, keep alpha constant and maximize with respect to c
			likevec.clear();
			for(int jq =0; jq < lenc; jq++){
				totallike = 0.0;
				for(auto it2 : resnormvec){
					totallike += exp(-robustcost(it2,c[jq], alpha[maxalphaind]))/constTable[maxalphaind][jq];
				}
				// std::cout << "Likelihood for  alpha = " << alpha[maxalphaind] << " and "<< " c = "<< c[jq] << " is " << totallike << endl;
				likevec.push_back(totallike);
			}

		    std::vector<double>::iterator result2;

		    result2 = std::max_element(likevec.begin(), likevec.end());
		    maxcind = std::distance(likevec.begin(), result2);
			std::cout << "Best c -- " << c[maxcind] << endl;
			// std::cout << " ------------------------ "  << std::endl;

			// thirdly, do iteratively re-weighted least squares
			int numIter = iteration_number_;
			if (corres_.size() < 10)
				return;

			std::vector<double> s(corres_.size(), 1.0);


			for (int itr2 = 0; itr2 < numIter; itr2++) {

				const int nvariable = 6;	// 3 for rotation and 3 for translation
				Eigen::MatrixXd JTJ(nvariable, nvariable);
				Eigen::MatrixXd JTr(nvariable, 1);
				Eigen::MatrixXd J(nvariable, 1);
				JTJ.setZero();
				JTr.setZero();

				double r;
				double r2 = 0.0;

				for (int cr = 0; cr < corres_.size(); cr++) {
					int ii = corres_[cr].first;
					int jj = corres_[cr].second;
					Eigen::Vector3f p, q;
					p = pointcloud_[i][ii];
					q = pcj_copy[jj];
					Eigen::Vector3f rpq = p - q;

					int c2 = cr;
					double res = rpq.norm();

					// weights of residuals derived using rho'(x)/x

					s[c2] = robustcostWeight(res, c[maxcind], alpha[maxalphaind]);

					J.setZero();
					J(1) = -q(2);
					J(2) = q(1);
					J(3) = -1;
					r = rpq(0);
					JTJ += J * J.transpose() * s[c2];
					JTr += J * r * s[c2];
					r2 += r * r * s[c2];

					J.setZero();
					J(2) = -q(0);
					J(0) = q(2);
					J(4) = -1;
					r = rpq(1);
					JTJ += J * J.transpose() * s[c2];
					JTr += J * r * s[c2];
					r2 += r * r * s[c2];

					J.setZero();
					J(0) = -q(1);
					J(1) = q(0);
					J(5) = -1;
					r = rpq(2);
					JTJ += J * J.transpose() * s[c2];
					JTr += J * r * s[c2];
					r2 += r * r * s[c2];

					// r2 += (par * (1.0 - sqrt(s[c2])) * (1.0 - sqrt(s[c2])));
				}

				Eigen::MatrixXd result(nvariable, 1);
				result = -JTJ.llt().solve(JTr);
				//std::cout << "Result is " << result << std::endl;

				Eigen::Affine3d aff_mat;
				aff_mat.linear() = (Eigen::Matrix3d) Eigen::AngleAxisd(result(2), Eigen::Vector3d::UnitZ())
					* Eigen::AngleAxisd(result(1), Eigen::Vector3d::UnitY())
					* Eigen::AngleAxisd(result(0), Eigen::Vector3d::UnitX());
				aff_mat.translation() = Eigen::Vector3d(result(3), result(4), result(5));

				Eigen::Matrix4f delta = aff_mat.matrix().cast<float>();
				trans = delta * trans;
				TransformPoints(pcj_copy, delta);
			}

			diff = (pretrans - trans).norm();
			std::cout << "Normed difference in trans -- " << diff << std::endl;
			std::cout << " ------------------------ " << std::endl;
		}
		else{
			break;
		}

	}

	TransOutput_ = trans * TransOutput_;
}

void CApp::TransformPoints(Points& points, const Eigen::Matrix4f& Trans)
{
	int npc = (int)points.size();
	Matrix3f R = Trans.block<3, 3>(0, 0);
	Vector3f t = Trans.block<3, 1>(0, 3);
	Vector3f temp;
	for (int cnt = 0; cnt < npc; cnt++) {
		temp = R * points[cnt] + t;
		points[cnt] = temp;
	}
}

Eigen::Matrix4f CApp::GetOutputTrans()
{
	Eigen::Matrix3f R;
	Eigen::Vector3f t;
	R = TransOutput_.block<3, 3>(0, 0);
	t = TransOutput_.block<3, 1>(0, 3);

	Eigen::Matrix4f transtemp;
	transtemp.fill(0.0f);

	transtemp.block<3, 3>(0, 0) = R;
	transtemp.block<3, 1>(0, 3) = -R*Means[1] + t*GlobalScale + Means[0];
	transtemp(3, 3) = 1;

	return transtemp;
}

void CApp::WriteTrans(const char* filepath)
{
	FILE* fid = fopen(filepath, "w");

	// Below line indicates how the transformation matrix aligns two point clouds
	// e.g. T * pointcloud_[1] is aligned with pointcloud_[0].
	// '2' indicates that there are two point cloud fragments.
	fprintf(fid, "0 1 2\n");

	Eigen::Matrix4f transtemp = GetOutputTrans();

	fprintf(fid, "%.10f %.10f %.10f %.10f\n", transtemp(0, 0), transtemp(0, 1), transtemp(0, 2), transtemp(0, 3));
	fprintf(fid, "%.10f %.10f %.10f %.10f\n", transtemp(1, 0), transtemp(1, 1), transtemp(1, 2), transtemp(1, 3));
	fprintf(fid, "%.10f %.10f %.10f %.10f\n", transtemp(2, 0), transtemp(2, 1), transtemp(2, 2), transtemp(2, 3));
	fprintf(fid, "%.10f %.10f %.10f %.10f\n", 0.0f, 0.0f, 0.0f, 1.0f);

	fclose(fid);
}

Eigen::Matrix4f CApp::ReadTrans(const char* filename)
{
	Eigen::Matrix4f temp;
	temp.fill(0);
	int temp0, temp1, temp2, cnt = 0;
	FILE* fid = fopen(filename, "r");
	while (fscanf(fid, "%d %d %d", &temp0, &temp1, &temp2) == 3)
	{
		for (int j = 0; j < 4; j++)
		{
			float a, b, c, d;
			fscanf(fid, "%f %f %f %f", &a, &b, &c, &d);
			temp(j, 0) = a;
			temp(j, 1) = b;
			temp(j, 2) = c;
			temp(j, 3) = d;
		}
	}
	return temp;
}

void CApp::BuildDenseCorrespondence(const Eigen::Matrix4f& trans,
		Correspondences& corres)
{
	int fi = 0;
	int fj = 1;
	Points pci = pointcloud_[fi];
	Points pcj = pointcloud_[fj];
	TransformPoints(pcj, trans);

	KDTree feature_tree_i(flann::KDTreeSingleIndexParams(15));
	BuildKDTree(pci, &feature_tree_i);
	std::vector<int> ind;
	std::vector<float> dist;
	corres.clear();
	for (int j = 0; j < pcj.size(); ++j)
	{
		SearchKDTree(&feature_tree_i, pcj[j], ind, dist, 1);
		float dist_j = sqrt(dist[0]);
		if (dist_j / GlobalScale < max_corr_dist_ / 2.0)
			corres.push_back(std::pair<int, int>(ind[0], j));
	}
}

void CApp::Evaluation(const char* gth, const char* estimation, const char *output)
{
	float inlier_ratio = -1.0f;
	float overlapping_ratio = -1.0f;

	int fi = 0;
	int fj = 1;

	std::vector<std::pair<int, int> > corres;
	Eigen::Matrix4f gth_trans = ReadTrans(gth);
	BuildDenseCorrespondence(gth_trans, corres);
	printf("Groundtruth correspondences [%d-%d] : %d\n", fi, fj,
			(int)corres.size());

	int ncorres = corres.size();
	float err_mean = 0.0f;

	Points pci = pointcloud_[fi];
	Points pcj = pointcloud_[fj];
	Eigen::Matrix4f est_trans = ReadTrans(estimation);
	std::vector<float> error;
	for (int i = 0; i < ncorres; ++i)
	{
		int idi = corres[i].first;
		int idj = corres[i].second;
		Eigen::Vector4f pi(pci[idi](0), pci[idi](1), pci[idi](2), 1);
		Eigen::Vector4f pj(pcj[idj](0), pcj[idj](1), pcj[idj](2), 1);
		Eigen::Vector4f pjt = est_trans*pj;
		float errtemp = (pi - pjt).norm();
		error.push_back(errtemp);
		// this is based on the RMSE defined in
		// https://en.wikipedia.org/wiki/Root-mean-square_deviation
		errtemp = errtemp * errtemp;
		err_mean += errtemp;
	}
	err_mean /= ncorres; // this is MSE = mean(d^2)
	err_mean = sqrt(err_mean); // this is RMSE = sqrt(MSE)
	printf("mean error : %0.4e\n", err_mean);

	//overlapping_ratio = (float)ncorres / min(
	//		pointcloud_[fj].size(), pointcloud_[fj].size());
	overlapping_ratio = (float)ncorres / pointcloud_[fj].size();

	// write errors
	FILE* fid = fopen(output, "w");
	fprintf(fid, "%d %d %e %e %e\n", fi, fj, err_mean,
			inlier_ratio, overlapping_ratio);
	fclose(fid);
}

double CApp::robustcost(double r, double c, double alpha){
	if (alpha == 2.0){
    	return 0.5*pow(r/c,2);}
	else if (alpha == 0.0){
    	return log(0.5*pow(r/c,2) + 1);}
	else if (alpha < -1000.0){
    	return 1 - exp(0.5*pow(r/c,2));}
	else {
    	return (abs(alpha-2)/alpha)*pow(r*r/(c*c*abs(alpha-2)) + 1,(alpha/2)-1);}

}
double CApp::robustcostWeight(double r, double c, double alpha){
	double weight;
	if (alpha == 2){
    	weight = 1;}
	else if (alpha == 0){
    	weight = 2*c*c/(r*r + 2*c*c);}
	else if (alpha < -1000){
    	weight = exp(-0.5*(r*r/c*c));}
	else {
    	weight = pow((r*r/(c*c*abs(alpha-2)) + 1),(alpha/2-1));}
	return weight;
}
