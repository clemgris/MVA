// Imagine++ project
// Project:  Fundamental
// Author:   Pascal Monasse

#include "./Imagine/Features.h"
#include <Imagine/Graphics.h>
#include <Imagine/LinAlg.h>
#include <vector>
#include <cstdlib>
#include <ctime>
using namespace Imagine;
using namespace std;

static const float BETA = 0.01f; // Probability of failure

struct Match {
    float x1, y1, x2, y2;
};

// Display SIFT points and fill vector of point correspondences
void algoSIFT(Image<Color,2> I1, Image<Color,2> I2,
              vector<Match>& matches) {
    // Find interest points
    SIFTDetector D;
    D.setFirstOctave(-1);
    Array<SIFTDetector::Feature> feats1 = D.run(I1);
    drawFeatures(feats1, Coords<2>(0,0));
    cout << "Im1: " << feats1.size() << flush;
    Array<SIFTDetector::Feature> feats2 = D.run(I2);
    drawFeatures(feats2, Coords<2>(I1.width(),0));
    cout << " Im2: " << feats2.size() << flush;

    const double MAX_DISTANCE = 100.0*100.0;
    for(size_t i=0; i < feats1.size(); i++) {
        SIFTDetector::Feature f1=feats1[i];
        for(size_t j=0; j < feats2.size(); j++) {
            double d = squaredDist(f1.desc, feats2[j].desc);
            if(d < MAX_DISTANCE) {
                Match m;
                m.x1 = f1.pos.x();
                m.y1 = f1.pos.y();
                m.x2 = feats2[j].pos.x();
                m.y2 = feats2[j].pos.y();
                matches.push_back(m);
            }
        }
    }
}

// Pick 8 different ids among [0, n[
void random_ids(int n, vector<int>& ids) {
    int id;
    while (ids.size() < 8) {
        id = intRandom(0, n-1);
        if (find(ids.begin(), ids.end(), id) == ids.end()) {
            ids.push_back(id);
        }
    }
}

// Compute the distance between X2 and the epipolar liner from X1 (F^T*X1)
// for the match (X1, X2)
float distance(FMatrix<float, 3, 3> F, Match match) {
    // Epipolar line from X1 = (x1, y1, 1)
    FVector<float, 3> X;
    FVector<float, 3> Xp;
    X[0] = match.x1;  X[1] = match.y1;  X[2] = 1.0f;
    Xp[0] = match.x2; Xp[1] = match.y2; Xp[2] = 1.0f;

    FVector<float, 3> epi_line = F * X;

    float d = abs(Xp * epi_line) / sqrt(epi_line[0] * epi_line[0] + epi_line[1] * epi_line[1]);

    return d;
}

// Compute F from a set of point matches
FMatrix<float, 3, 3> modelF(vector<Match> matches, bool forceRank2 = true) {

    assert (matches.size() >= 8);

    // Normalize
    FMatrix<float, 3, 3> N;
    float norm_factor = 0.001;

    N(0, 0) = norm_factor; N(0, 1) = 0;           N(0, 2) = 0;
    N(1, 0) = 0;           N(1, 1) = norm_factor; N(1, 2) = 0;
    N(2, 2) = 0;           N(2, 1) = 0;           N(2, 2) = 1;

    // Compute matrix A st A*f = 0
    Matrix<double> A(matches.size(), 9);
    for (size_t id = 0; id < matches.size(); id++) {
        Match match = matches[id];
        A(id, 0) = (norm_factor * norm_factor) * match.x1 * match.x2;
        A(id, 1) = (norm_factor * norm_factor) * match.y1 * match.x2;
        A(id, 2) = norm_factor * match.x2;
        A(id, 3) = (norm_factor * norm_factor) * match.x1 * match.y2;
        A(id, 4) = (norm_factor * norm_factor) * match.y1 * match.y2;
        A(id, 5) = norm_factor * match.y2;
        A(id, 6) = norm_factor * match.x1;
        A(id, 7) = norm_factor * match.y1;
        A(id, 8) = 1.0f;
    }

    // SVD decomposition of A
    Vector<double> S(9);
    Matrix<double> U(matches.size(), matches.size());
    Matrix<double> Vt(9,9);
    svd(A, U, S, Vt);

    // Take f = V_9 eigenvector associated with smallest eigenvalue of A
    FMatrix<float, 3, 3> F;
    F(0, 0) = Vt(8, 0); F(0, 1) = Vt(8, 1); F(0, 2) = Vt(8, 2);
    F(1, 0) = Vt(8, 3); F(1, 1) = Vt(8, 4); F(1, 2) = Vt(8, 5);
    F(2, 0) = Vt(8, 6); F(2, 1) = Vt(8, 7); F(2, 2) = Vt(8, 8);

    // Denormalize
    F = transpose(N) * F * N;

    if (forceRank2) {
        // Impose rank(F)=2
        FVector<float, 3> S_f;
        FMatrix<float, 3, 3> U_f;
        FMatrix<float, 3, 3> Vt_f;
        svd(F, U_f, S_f, Vt_f);
        S_f[2] = 0.0f;

        F = U_f * Diagonal(S_f) * Vt_f;
    }

    return F;
}

// RANSAC algorithm to compute F from point matches (8-point algorithm)
// Parameter matches is filtered to keep only inliers as output.
FMatrix<float,3,3> computeF(vector<Match>& matches) {
    const float distMax = 1.5f; // Pixel error for inlier/outlier discrimination
    int Niter = 100000; // Adjusted dynamically
    FMatrix<float, 3, 3> bestF;
    vector<int> bestInliers;

    int N = 0;
    FMatrix<float, 3, 3> F;
    vector<int> eight_ids;
    vector<Match> eight_matches;
    vector<int> inliers;

    while (N < Niter) {

        // Select 8 random point matches
        eight_ids.clear();
        eight_matches.clear();
        random_ids((int)matches.size(), eight_ids);
        for (int id : eight_ids) {
            eight_matches.push_back(matches[id]);
        }

        // Compute F
        F = modelF(eight_matches, false);

        // Divide inliers/outliers
        inliers.clear();
        for (size_t i = 0; i < matches.size(); i ++) {
            if (distance(F, matches[i]) <= distMax) {
                inliers.push_back(i);
            }
        }

        // Update best model
        if (inliers.size() > bestInliers.size()) {
            bestInliers = inliers;
            bestF = F;
            // Avoid division by 0
            if (abs(log(1-pow((float)bestInliers.size() / (float)matches.size(), 8))) > abs(log(BETA)) / 100000) {
                Niter = ceil(log(BETA)/ log(1-pow((float)bestInliers.size() / (float)matches.size(), 8)));
            }
        }
        N++;
    }

    // Updating matches with inliers only
    vector<Match> all=matches;
    matches.clear();
    for(size_t i=0; i<bestInliers.size(); i++)
        matches.push_back(all[bestInliers[i]]);

    // Compute F with all the inliers
    bestF = modelF(matches);

    return bestF;
}

// Expects clicks in one image and show corresponding line in other image.
// Stop at right-click.
void displayEpipolar(Image<Color> I1, Image<Color> I2,
                     const FMatrix<float,3,3>& F) {
    while(true) {
        int x,y;
        if (getMouse(x,y) == 3)
            break;
        FVector<float, 3> X;

        if (x < I1.width()) {

            X[0] = x;  X[1] = y;  X[2] = 1.0f;

            // Epipolar line from X1 = (x, y , 1)
            FVector<float, 3> epi_line = F * X;

            IntPoint2 pt1p(I1.width(), epi_line[2] / ( - epi_line[1]));
            IntPoint2 pt2p(I1.width() + I2.width(), (epi_line[0] * I2.width() + epi_line[2]) / ( - epi_line[1]));
            drawLine(pt1p, pt2p, RED);
        } else {
            X[0] = x - I1.width();  X[1] = y;  X[2] = 1.0f;

            // Epipolar line from X2 = (x, y , 1)
            FVector<float, 3> epi_line = transpose(F) * X;

            IntPoint2 pt1(0, epi_line[2] / ( - epi_line[1]));
            IntPoint2 pt2(I1.width(), (epi_line[0] * I1.width() + epi_line[2]) / ( - epi_line[1]));
            drawLine(pt1, pt2, MAGENTA);
        }
    }
}


int main(int argc, char* argv[])
{
    srand((unsigned int)time(0));

    const char* s1 = argc>1? argv[1]: srcPath("im1.jpg");
    const char* s2 = argc>2? argv[2]: srcPath("im2.jpg");

    // Load and display images
    Image<Color,2> I1, I2;
    if( ! load(I1, s1) ||
        ! load(I2, s2) ) {
        cerr<< "Unable to load images" << endl;
        return 1;
    }
    int w = I1.width();
    openWindow(2*w, I1.height());
    display(I1,0,0);
    display(I2,w,0);

    vector<Match> matches;
    algoSIFT(I1, I2, matches);
    const int n = (int)matches.size();
    cout << " matches: " << n << endl;
    drawString(100,20,std::to_string(n)+ " matches",RED);
    click();
    
    FMatrix<float,3,3> F = computeF(matches);
    cout << "F="<< endl << F;

    // Redisplay with matches
    display(I1,0,0);
    display(I2,w,0);
    for(size_t i=0; i<matches.size(); i++) {
        Color c(rand()%256,rand()%256,rand()%256);
        fillCircle(matches[i].x1+0, matches[i].y1, 2, c);
        fillCircle(matches[i].x2+w, matches[i].y2, 2, c);        
    }
    drawString(100, 20, to_string(matches.size())+"/"+to_string(n)+" inliers", RED);
    click();

    // Redisplay without SIFT points
    display(I1,0,0);
    display(I2,w,0);
    displayEpipolar(I1, I2, F);

    endGraphics();
    return 0;
}
