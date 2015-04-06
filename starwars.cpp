#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <list>
#include <cstdio>

using namespace cv;
using namespace std;

void turnPic(Mat& p0, Mat& p, double phi)
{
  int newSz = (int)sqrt(p0.rows*p0.rows + p0.cols*p0.cols);
  Mat p1(newSz,newSz,CV_8UC4);
  p1.setTo(0);
  int offsetX = (newSz - p0.cols) / 2;
  int offsetY = (newSz - p0.rows) / 2;
  p0.copyTo(p1.rowRange(offsetY, offsetY + p0.rows).colRange(offsetX, offsetX + p0.cols));
  Point2f center(newSz/2.0F, newSz/2.0F);
  Mat rot_mat = getRotationMatrix2D(center, phi, 1.0);
  warpAffine(p1, p, rot_mat, p1.size());
}

void overlayWithTransparancy(Mat &mf, Mat &f)
{
  for(int i = 0; i < f.rows; i++)
  {
    Vec3b* mp = mf.ptr<Vec3b>(i); //pointer to 1st pixel in row
    Vec4b* p = f.ptr<Vec4b>(i); //pointer to 1st pixel in row
    for (int j = 0; j < f.cols; j++)
    {
      double t = p[j][3]/255.0;
      for (int ch = 0; ch < 3; ch++)
      {
        double mpNew = mp[j][ch]*(1-t) + p[j][ch]*t;
        mp[j][ch] = (uchar)max(0.0,min(mpNew,255.0));
      }
    }
  }
}

struct Ship {
  Mat pic, pic0;
  double x,y,vx,vy,phi, scale; //x,y - позиция центра
  void init(Mat& asteroidPic, double x0, double y0, double v1, double v2, double phi0, double scale)
  {
    resize(asteroidPic, pic0, Size(scale*asteroidPic.cols, scale*asteroidPic.rows));
    x = x0;  y = y0;  vx = v1; vy = v2; phi = phi0;
    turnPic(pic0, pic, phi);
  }
  void move(double dt)
  {
    x += vx*dt;  y -= vy*dt;
    turnPic(pic0, pic, phi);
  }
  void display(Mat& mainFrame)
  {
    Mat frame(Size(mainFrame.cols,mainFrame.rows), CV_8UC4);
    frame.setTo(0);
    int offsetX = x-pic.cols/2, offsetY = y-pic.rows/2;

    //intersect region in frame
    Rect roiFrame( Point( offsetX, offsetY ), Size( pic.cols, pic.rows ));
    roiFrame.x = max(roiFrame.x,0); roiFrame.y = max(roiFrame.y,0);
    if (offsetX<0) roiFrame.width += offsetX;
    if (offsetY<0) roiFrame.height += offsetY;
    roiFrame.height = min(roiFrame.height,frame.rows-roiFrame.y);
    roiFrame.width = min(roiFrame.width,frame.cols-roiFrame.x);

    //intersect region in a.pic
    Rect roia( Point(0,0), Size( roiFrame.width, roiFrame.height ));
    if (offsetX<0) roia.x -= offsetX;
    if (offsetY<0) roia.y -= offsetY;

    if (roia.height<=0 || roia.width<=0)   return;
    pic(roia).copyTo( frame(roiFrame) );
    overlayWithTransparancy(mainFrame, frame);
  }
  void changeSpeed(double mx, double my)
  {
    double q1 = mx-x, q2 = -my+y, q=sqrt(q1*q1+q2*q2);
    double dv = min(q,100.0);
    vx = q1/q*dv; vy = q2/q*dv;
    cout << "Changing speed: vx= " << vx <<" vy=" << vy << " phi = " << phi<< endl;
  }
} ship;

double fRand(double fMin, double fMax)
{
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

string getImgType(Mat& im)
{
  int enum_ints[] =       {CV_8U,  CV_8UC1,  CV_8UC2,  CV_8UC3,  CV_8UC4,
                             CV_8S,  CV_8SC1,  CV_8SC2,  CV_8SC3,  CV_8SC4,
                             CV_16U, CV_16UC1, CV_16UC2, CV_16UC3, CV_16UC4,
                             CV_16S, CV_16SC1, CV_16SC2, CV_16SC3, CV_16SC4,
                             CV_32S, CV_32SC1, CV_32SC2, CV_32SC3, CV_32SC4,
                             CV_32F, CV_32FC1, CV_32FC2, CV_32FC3, CV_32FC4,
                             CV_64F, CV_64FC1, CV_64FC2, CV_64FC3, CV_64FC4};
  string enum_strings[] = {"CV_8U",  "CV_8UC1",  "CV_8UC2",  "CV_8UC3",  "CV_8UC4",
                             "CV_8S",  "CV_8SC1",  "CV_8SC2",  "CV_8SC3",  "CV_8SC4",
                             "CV_16U", "CV_16UC1", "CV_16UC2", "CV_16UC3", "CV_16UC4",
                             "CV_16S", "CV_16SC1", "CV_16SC2", "CV_16SC3", "CV_16SC4",
                             "CV_32S", "CV_32SC1", "CV_32SC2", "CV_32SC3", "CV_32SC4",
                             "CV_32F", "CV_32FC1", "CV_32FC2", "CV_32FC3", "CV_32FC4",
                             "CV_64F", "CV_64FC1", "CV_64FC2", "CV_64FC3", "CV_64FC4"};
  int imgTypeInt = im.type();
  for(int i=0; i<sizeof(enum_ints)/sizeof(int); i++)
  {
    if(imgTypeInt == enum_ints[i]) return enum_strings[i];
  }
  return "unknown image type";
}

double fRand(double fMax) {return fRand(0,fMax);}

void setTransparancyTo(Mat& frame, uchar v)
{
  Mat channel[4];
  split(frame, channel);
  channel[3].setTo(v);
  merge(channel,4,frame);
}

void on_mouse(int event, int x, int y, int flags, void* userdata)
{
  if ( event == EVENT_MOUSEMOVE )
  {
    double c = x-ship.x, s = -y+ship.y, r = sqrt(c*c+s*s);
    if (r <1e-2) return;
    c /= r; s /= r;
    double phi;
    if (fabs(c)<1e-2) phi = s>0 ? 90 : -90;
    else phi = atan2(s,c)/3.14*180;
    ship.phi = phi;
    ship.changeSpeed(x,y);
  }
}

int main( int argc, char** argv )
{
  VideoCapture cap("StarWars.avi");
  if (!cap.isOpened()) { cout << "Error can't find the file"<<endl; }
  Mat shipPic = imread("ship.png", -1);
  Mat mainFrame;

  ship.init(shipPic, 320, 240, 0,0, 90, 0.5);

  namedWindow( "Star wars", WINDOW_AUTOSIZE );// Create a window for display.
  setMouseCallback("Star wars", on_mouse, 0 );
  int fr = 0;
  while (true)
  {
    if (!cap.read(mainFrame)) break;
    ship.move(1.0/24);
    ship.display(mainFrame);
    char text[255];
    sprintf(text, "Score:%d", (int)fr);
    putText (mainFrame, text, Point(500,20), CV_FONT_HERSHEY_SIMPLEX|CV_FONT_ITALIC, 0.7, Scalar(255,255,255), 2);
    imshow("Star wars", mainFrame); //show the frame in "MyVideo" window
    waitKey(20);
    fr++;
  }
  waitKey(0);
  return 0;
}
