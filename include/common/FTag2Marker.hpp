#ifndef FTAG2MARKER_HPP_
#define FTAG2MARKER_HPP_

struct FTag2Marker {

  unsigned int ID;

  double pose_x;
  double pose_y;
  double pose_z;

  double orientation_x;
  double orientation_y;
  double orientation_z;
  double orientation_w;

  FTag2Marker() : ID(0) {};
};


#endif /* FTAG2MARKER_HPP_ */
