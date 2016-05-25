# virtualnav-mpu6050
Navigation in a Virtual Environment using IMU MPU-6050 with live filtered graph output for gyro and accel_y axis.

Head Mounted Display(HMD) is one of the revolutionary Virtual Reality(VR) inventions of all times. But how do you move around in a Virtual Environment?. For a true VR experience you need to move around freely and naturally. Imagine a game where the user can freely roam around their backyard or walk on a frictionless surface and navigate in a virtual environment rather than sitting idle in a chair. Similarly there are a ton of applications from taking a morning walk in a VR world to a whole range of simulation training for combact soldiers. Developing a low cost system for such a VR experience which can be implemented onto a HMD, is always a challenge. In this project we have done a hardware implementation to navigate in a virtual environment using a low cost Inertial Measurement Unit(IMU).

The UDP client is implemented in the Game side, implement the server similarly from the Pi side for the game to work. You can go through this tutorial to read the data from IMU MPU-6050.
http://blog.bitify.co.uk/2013/11/reading-data-from-mpu-6050-on-raspberry.html
