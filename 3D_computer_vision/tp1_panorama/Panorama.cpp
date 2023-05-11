// Imagine++ project
// Project:  Panorama
// Author:   Pascal Monasse
// Date:     2013/10/08

#include <Imagine/Graphics.h>
#include <Imagine/Images.h>
#include <Imagine/LinAlg.h>
#include <vector>
#include <sstream>
using namespace Imagine;
using namespace std;

// Record clicks in two images, until right button click
void getClicks(Window w1,
               Window w2,
               vector<IntPoint2>& pts1,
               vector<IntPoint2>& pts2,
               int max_n = 100) {
    int click = 0;
    IntPoint2 pt1;
    IntPoint2 pt2;
    for (int i = 0; i < max_n; i++) {
        setActiveWindow(w1);
        click = getMouse(pt1.x(), pt1.y());
        // Check for left click
        if (click == 3) break;

        setActiveWindow(w2);
        click = getMouse(pt2.x(), pt2.y());
        // Check for left click
        if (click == 3) break;

        pts1.push_back(pt1);
        pts2.push_back(pt2);
    }
}

// Return homography compatible with point matches
Matrix<float> getHomography(const vector<IntPoint2>& pts1,
                            const vector<IntPoint2>& pts2) {
    size_t n = min(pts1.size(), pts2.size());
    if(n<4) {
        cout << "Not enough correspondences: " << n << endl;
        return Matrix<float>::Identity(3);
    }
    Matrix<double> A(2*n, 8);
    Vector<double> B(2*n);

    // Solve linear system A(:,1:8)*h(1:8) = -A(:,9)
    for (size_t i = 0; i<n; i++) {
        A(2*i, 0) = pts1[i].x();
        A(2*i, 1) = pts1[i].y();
        A(2*i, 2) = 1.0;
        A(2*i, 3) = 0.0;
        A(2*i, 4) = 0.0;
        A(2*i, 5) = 0.0;
        A(2*i, 6) = -pts2[i].x()*pts1[i].x();
        A(2*i, 7) = -pts2[i].x()*pts1[i].y();

        A(2*i+1, 0) = 0.0;
        A(2*i+1, 1) = 0.0;
        A(2*i+1, 2) = 0.0;
        A(2*i+1, 3) = pts1[i].x();
        A(2*i+1, 4) = pts1[i].y();
        A(2*i+1, 5) = 1.0;
        A(2*i+1, 6) = -pts2[i].y()*pts1[i].x();
        A(2*i+1, 7) = -pts2[i].y()*pts1[i].y();

        B[2*i] = pts2[i].x();
        B[2*i+1] = pts2[i].y();
   }
    B = linSolve(A, B);
    Matrix<float> H(3, 3);
    H(0,0)=B[0]; H(0,1)=B[1]; H(0,2)=B[2];
    H(1,0)=B[3]; H(1,1)=B[4]; H(1,2)=B[5];
    H(2,0)=B[6]; H(2,1)=B[7]; H(2,2)=1;

    // Sanity check
    for(size_t i=0; i<n; i++) {
        float v1[]={(float)pts1[i].x(), (float)pts1[i].y(), 1.0f};
        float v2[]={(float)pts2[i].x(), (float)pts2[i].y(), 1.0f};
        Vector<float> x1(v1,3);
        Vector<float> x2(v2,3);
        x1 = H*x1;
        cout << x1[1]*x2[2]-x1[2]*x2[1] << ' '
             << x1[2]*x2[0]-x1[0]*x2[2] << ' '
             << x1[0]*x2[1]-x1[1]*x2[0] << endl;
    }
    return H;
}

// Grow rectangle of corners (x0,y0) and (x1,y1) to include (x,y)
void growTo(float& x0,
            float& y0,
            float& x1,
            float& y1,
            float x,
            float y) {
    if(x<x0) x0=x;
    if(x>x1) x1=x;
    if(y<y0) y0=y;
    if(y>y1) y1=y;    
}

// Panorama construction
void panorama(const Image<Color,2>& I1,
              const Image<Color,2>& I2,
              Matrix<float> H) {
    Vector<float> v(3);
    float x0=0, y0=0, x1=I2.width(), y1=I2.height();

    v[0]=0; v[1]=0; v[2]=1;
    v=H*v; v/=v[2];
    growTo(x0, y0, x1, y1, v[0], v[1]);

    v[0]=I1.width(); v[1]=0; v[2]=1;
    v=H*v; v/=v[2];
    growTo(x0, y0, x1, y1, v[0], v[1]);

    v[0]=I1.width(); v[1]=I1.height(); v[2]=1;
    v=H*v; v/=v[2];
    growTo(x0, y0, x1, y1, v[0], v[1]);

    v[0]=0; v[1]=I1.height(); v[2]=1;
    v=H*v; v/=v[2];
    growTo(x0, y0, x1, y1, v[0], v[1]);

    cout << "x0 x1 y0 y1=" << x0 << ' ' << x1 << ' ' << y0 << ' ' << y1<<endl;

    Image<Color> I(int(x1-x0), int(y1-y0));
    setActiveWindow( openWindow(I.width(), I.height()) );
    I.fill(WHITE);

    Matrix<float> inverse_H = inverse(H);
    Vector<float> w(3);

    for (int x = int(x0); x < int(x1); x++) {
        for (int y = int(y0); y < int(y1); y++) {

            // Pull
            w[0] = x;
            w[1]=y;
            w[2]=1.0f;
            w = inverse_H*w;
            w/=w[2];
            int x_p = int(w[0]);
            int y_p = int(w[1]);

            bool in_I2 = (0 <= x && x < I2.width() && 0 <= y && y < I2.height());
            bool in_HI1 = (0 <= x_p && x_p < I1.width() && 0 <= y_p && y_p < I1.height());

            if (in_HI1) {
               if (in_I2) {
                   // Pixel in I2 and HI1 --> mean of the two intensities
                   I(x-int(x0), y-int(y0)) = Color((I1(x_p, y_p).r() + I2(x, y).r())/2,
                                                   (I1(x_p, y_p).g() + I2(x, y).g())/2,
                                                   (I1(x_p, y_p).b() + I2(x, y).b())/2);
               } else {
                   // Pixel only in HI1
                   I(x-int(x0), y-int(y0)) = I1(x_p, y_p);
               }
            } else if (in_I2) {
                // Pixel only in I2
                I(x-int(x0), y-int(y0)) = I2(x, y);
            }
        }
    }
    display(I,0,0);
}

// Main function
int main(int argc, char* argv[]) {
    const char* s1 = argc>1? argv[1]: srcPath("image0006.jpg");
    const char* s2 = argc>2? argv[2]: srcPath("image0007.jpg");

    // Load and display images
    Image<Color> I1, I2;
    if( ! load(I1, s1) ||
        ! load(I2, s2) ) {
        cerr<< "Unable to load the images" << endl;
        return 1;
    }
    Window w1 = openWindow(I1.width(), I1.height(), s1);
    display(I1,0,0);
    Window w2 = openWindow(I2.width(), I2.height(), s2);
    setActiveWindow(w2);
    display(I2,0,0);

    // Get user's clicks in images
    vector<IntPoint2> pts1, pts2;

    cout << "enter fonction" << endl;
    getClicks(w1, w2, pts1, pts2);

    vector<IntPoint2>::const_iterator it;
    cout << "pts1="<<endl;
    for(it=pts1.begin(); it != pts1.end(); it++)
        cout << *it << endl;
    cout << "pts2="<<endl;
    for(it=pts2.begin(); it != pts2.end(); it++)
        cout << *it << endl;

    // Compute homography
    Matrix<float> H = getHomography(pts1, pts2);
    cout << "H=" << H/H(2,2);

    // Apply homography
    panorama(I1, I2, H);

    endGraphics();
    return 0;
}
