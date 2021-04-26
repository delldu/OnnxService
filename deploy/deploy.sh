#/************************************************************************************#***
#***	Copyright Dell 2021, All Rights Reserved.
#***
#***	File Author: Dell, 2021年 04月 21日 星期三 13:36:22 CST
#***
#************************************************************************************/
#
#! /bin/sh

# temp share with docker building

# software-properties-common


help()
{
	echo "before docker after ..."
}

before()
{
	echo "Building prepare ..."

	rm -rf temp && mkdir temp
	cp /opt/onnxruntime-linux-x64-gpu-1.6.0 temp/onnxruntime -R
}

docker()
{
	echo "Building deployment docker ..."
	docker build -t "onnxservice:runner" .
}

after()
{
	echo "Building clean ..."

	rm -rf temp
}

if [ "$*" == "" ] ;
then
	help
else
	eval "$*"
fi

