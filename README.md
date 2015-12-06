## Install OpenCV (latest) for RHEL
```Shell
# inspired by http://superuser.com/questions/678568/install-opencv-in-centos
sudo yum groupinstall "Development Tools"
sudo yum install gcc cmake git gtk2-devel pkgconfig numpy ffmpeg
sudo mkdir /opt/working
cd /opt/working
sudo git clone https://github.com/Itseez/opencv.git
cd opencv
sudo mkdir release
cd release
sudo cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local ..
sudo make
sudo make install
```

## Configure OpenCV build paths
1. Add OpenCV package config: `$ touch /usr/local/lib/pkgconfig/opencv.pc`
2. Add the following lines to `.bashrc` and `source` it: 
```
PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/usr/local/lib/pkgconfig
export PKG_CONFIG_PATH
```
3. Add the following flags to the package config:
```Shell
$ pkg-config --cflags opencv
$ pkg-config --libs opencv
```

## Compile OpenCV code
To compile any `.cpp` OpenCV program, use the `Makefile` like:
`$ make <program_name_without_extension>`

