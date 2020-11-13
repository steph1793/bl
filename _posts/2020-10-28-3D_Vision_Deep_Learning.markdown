---
layout: post
title:  "Deep Learning for 3D Point Clouds - Part I (Introduction)"
date:   2020-10-28
description: A walkthrough 3D LIDAR point clouds processing in computer vision with Deep Learning methods for semantic segmentation. An in-depth analysis of some architectures and an introduction to Neural Architecture Search in 3D Computer Vision.
---


<p style="text-align: justify; font-size: 0.9em"><span class="dropcap">T</span>he development of 3D data acquisition technologies like LIDAR or RGB-D cameras, the increasing computing capacity and the recent breakthroughs in Deep Learning has favoured the use of 3D data in a variety of fields for perception : robotics, medical imaging, self-driving cars etc. Those data can be represented as point clouds, RGB-D images or meshes. Despite 2D images, 3D data provide a depth information.
<br>In robotics and self-driving cars, perception is crucial. A robot necessarily needs to localize the wrist of a glass to be able to grab it; an autonomous vehicle needs to detect the other vehicles, the traffic lights, the road, to be able to drive. This necessity has given rise to a number of tasks in 3D data processing like object detection, semantic segmentation, instance segmentation for scene understanding, object part segmentation, object classification <b>[1]</b>.
</p>

### Introduction
<p style="text-align: justify; font-size: 0.9em">A point cloud is a set of un-ordered points lying in a three-dimensional space. Each point can be associated with features like a RGB color, a normal vector, an intensity etc. Point clouds are generally acquired with sensors like LIDARs. 
Throughout the years, numerous Deep Learning methods have emerged for point clouds processing. Here are some of the main approaches.</p>


<p style="text-align: justify; font-size: 0.9em"><b>Multi-View based methods</b><br>In such methods <b>[2]</b>, <b>[3]</b>, <b>[4]</b>, the 3D point clouds are projected onto 2D images from different angles. A 2D convolutional neural network (CNNs) <b>[5]</b> is then leveraged to extract some features from those images which are further processed for tasks like  classification and even 3D segmentation.</p>



<p style="text-align: justify; font-size: 0.9em"> <b>3D Volumetric Convolution (voxel-based) methods</b><br>They are drawn from the same motivations behind 2D CNNs : learning local patterns. In those methods, 3-dimensional kernels are used to apply volumetric convolutions directly in the 3D space. However, given the irregularity of the point clouds, such 3D convolutions cannot be directly applied to  the raw points. The 3D space is first voxelized into a regular grid, where each cell is coded in a binary way (empty or not) <b>[6]</b>; or represented by an aggregated vector of the points falling in this very same cell (the null vector if the cell is empty) <b>[7]</b>.</p>


<p style="text-align: justify; font-size: 0.9em"> <b>Graph-based methods</b><br>
In this category, the point clouds are represented with a graph structure. <b>[10]</b> proposes a method called edge-convolution where a multilayer perceptron MLP is applied to the neighbors of each node followed by an aggregation layer to build a feature vector for each node. A global max pooling can be applied to the feature vectors to extract a global encoding of the entire point cloud for classification.</p>


<p style="text-align: justify; font-size: 0.9em"> <b>Point based methods</b><br>
Point based methods take as input the raw point clouds as un-ordered sets of points. <b>[11]</b>, <b>[12]</b> leverage MLPs and max pooling to build a symetric function able to process the raw point clouds and extract some features per point or a global feature encoding the entire point cloud. On the other hand, <b>[13]</b>, <b>[14]</b> directly apply convolution operations on the points without any voxelization of the space, to learn local patterns. </p>


The list above is not exhaustive. There are many other families of Deep Learning methods like <b>[15]</b> where a voxel representation + raw point cloud representation are considered as input. Multi-modal approaches consider multiple data sources like LIDAR point clouds + RGB-D images as in <b>[16]</b>. <b>[17]</b>, <b>[18]</b> propose sparse convolutional networks, where the space is voxelized into a sparse 3D grid. The 3D voxel convolutions are applied only on the non-empty cells which sometimes can represent less than 10% of the voxels grid <b>[7]</b>.

The previous lines were a brief overview of Deep Learning in point clouds processing. A broader view of these families of approaches is exposed in <b>[19]</b>. In this article, let us put the focus on some specific architectures. In the last paper that we will study, the authors used <b>Neural Architecture Search (NAS)</b> to find the best design of their neural architecture.


### Table of Contents

* <a href="{% link _posts/2020-10-28-3D_point_based.markdown %}">MLP Point based methods</a>
  * <a href="{% link _posts/2020-10-28-3D_point_based.markdown %}"> PointNet</a>
  * <a href="{% link _posts/2020-10-28-3D_point_based.markdown %}"> PointNet++</a>
* <a href="{% link _posts/2020-11-08-3D_point_convolution.markdown %}"> Convolution Point Based methods</a>
  * <a href="{% link _posts/2020-11-08-3D_point_convolution.markdown %}"> PointCNN</a>
  * <a href=""> KPConv</a>
* <a href="{% link _posts/2020-11-08-3D_pointvoxel_based.markdown %}"> Point-Voxel Based methods</a>
  * <a href="{% link _posts/2020-11-08-3D_pointvoxel_based.markdown %}"> PVCNN</a>
* <a href="{% link _posts/2020-11-08-3D_mvpnet.markdown %}"> Multi-Modal Based methods</a>
  * <a href="{% link _posts/2020-11-08-3D_mvpnet.markdown %}"> MVPNet</a>
* <a href=""> Sparse Convolution Based methods</a>
  * <a href=""> Minkowski</a>
  * <a href=""> SPVCNN/SPVNAS</a>


<br>


### References
<br>

<textarea id="bibtex_input" style="display:none;">
@misc{1,
      title={Deep Learning for 3D Point Clouds: A Survey}, 
      author={Yulan Guo and Hanyun Wang and Qingyong Hu and Hao Liu and Li Liu and Mohammed Bennamoun},
      year={2020},
      eprint={1912.12033},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      pos={1}
}
}

@inproceedings{2,
  title={Multi-view convolutional neural networks for 3d shape recognition},
  author={Su, Hang and Maji, Subhransu and Kalogerakis, Evangelos and Learned-Miller, Erik},
  booktitle={Proceedings of the IEEE international conference on computer vision},
  pages={945--953},
  year={2015},
  pos={2}
}
@article{3,
  title={3D object retrieval with stacked local convolutional autoencoder},
  author={Leng, Biao and Guo, Shuang and Zhang, Xiangyang and Xiong, Zhang},
  journal={Signal Processing},
  volume={112},
  pages={119--128},
  year={2015},
  publisher={Elsevier},
  pos={3}
}

@inproceedings{4,
  title={Gift: A real-time and scalable 3d shape search engine},
  author={Bai, Song and Bai, Xiang and Zhou, Zhichao and Zhang, Zhaoxiang and Jan Latecki, Longin},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={5023--5032},
  year={2016},
  pos={4}
}

@article{5,
  title={Imagenet classification with deep convolutional neural networks},
  author={Krizhevsky, Alex and Sutskever, Ilya and Hinton, Geoffrey E},
  journal={Communications of the ACM},
  volume={60},
  number={6},
  pages={84--90},
  year={2017},
  publisher={ACM New York, NY, USA},
  pos={5}
}

@INPROCEEDINGS{6,
  author={D. {Maturana} and S. {Scherer}},
  booktitle={2015 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, 
  title={VoxNet: A 3D Convolutional Neural Network for real-time object recognition}, 
  year={2015},
  volume={},
  number={},
  pages={922-928},
  doi={10.1109/IROS.2015.7353481},
  pos={6}
}

@INPROCEEDINGS{7,
  author={Y. {Zhou} and O. {Tuzel}},
  booktitle={2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition}, 
  title={VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection}, 
  year={2018},
  volume={},
  number={},
  pages={4490-4499},
  doi={10.1109/CVPR.2018.00472},
  pos={7}
}

@article{8,
  title={Dynamic graph cnn for learning on point clouds},
  author={Wang, Yue and Sun, Yongbin and Liu, Ziwei and Sarma, Sanjay E and Bronstein, Michael M and Solomon, Justin M},
  journal={Acm Transactions On Graphics (tog)},
  volume={38},
  number={5},
  pages={1--12},
  year={2019},
  publisher={ACM New York, NY, USA},
  pos={10}
}

@INPROCEEDINGS{9,
author={R. Q. {Charles} and H. {Su} and M. {Kaichun} and L. J. {Guibas}},
booktitle={2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}, title={PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation},
year={2017},
volume={},
number={},
pages={77-85},
doi={10.1109/CVPR.2017.16},
pos={11}}
}

@inproceedings{10++,
  title={Pointnet++: Deep hierarchical feature learning on point sets in a metric space},
  author={Qi, Charles Ruizhongtai and Yi, Li and Su, Hao and Guibas, Leonidas J},
  booktitle={Advances in neural information processing systems},
  pages={5099--5108},
  year={2017},
  pos={12}
}

@INPROCEEDINGS{11,  author={H. {Thomas} and C. R. {Qi} and J. {Deschaud} and B. {Marcotegui} and F. {Goulette} and L. {Guibas}},  booktitle={2019 IEEE/CVF International Conference on Computer Vision (ICCV)},   title={KPConv: Flexible and Deformable Convolution for Point Clouds},   year={2019},  volume={},  number={},  pages={6410-6419},  doi={10.1109/ICCV.2019.00651}, pos={13}}

@incollection{12,
title = {PointCNN: Convolution On X-Transformed Points},
author = {Li, Yangyan and Bu, Rui and Sun, Mingchao and Wu, Wei and Di, Xinhan and Chen, Baoquan},
booktitle = {Advances in Neural Information Processing Systems 31},
editor = {S. Bengio and H. Wallach and H. Larochelle and K. Grauman and N. Cesa-Bianchi and R. Garnett},
pages = {820--830},
year = {2018},
publisher = {Curran Associates, Inc.},
pos={14}
}

@inproceedings{13,
  title={Point-Voxel CNN for Efficient 3D Deep Learning},
  author={Zhijian Liu and Haotian Tang and Yujun Lin and Song Han},
  booktitle={NeurIPS},
  year={2019},
  pos={15}
}

@inproceedings{14,
  title={Multi-view pointnet for 3d scene understanding},
  author={Jaritz, Maximilian and Gu, Jiayuan and Su, Hao},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision Workshops},
  pages={0--0},
  year={2019},
  pos={16}
}

@misc{15,
      title={4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks}, 
      author={Christopher Choy and JunYoung Gwak and Silvio Savarese},
      year={2019},
      eprint={1904.08755},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      pos={17}
}

 @inproceedings{16,
    title     = {Searching Efficient 3D Architectures with Sparse Point-Voxel Convolution},
    author    = {Tang, Haotian* and Liu, Zhijian* and Zhao, Shengyu and Lin, Yujun and Lin, Ji and Wang, Hanrui and Han, Song},
    booktitle = {European Conference on Computer Vision},
    year      = {2020},
    pos={18}
 } 

 @article{17,
 	title={Review: Deep Learning on 3D Point Clouds},
 	author={Bello Saifullahi A. and Yu Shangshu and Wang Cheng and Adam Jibril M. and Li Jonathan},
 	journal={Remote Sensing},
 	year={2020},
 	pos={19}
}

</textarea>

<div class="bibtex_template">
	<table style="border: none; margin-top: -30px;">
		<td style="vertical-align:top; border:none; width: 50px;"> [<span class="pos"></span>]
		</td>
	<td>
	
  <div class="if author" style="font-weight: bold;">	
	<div >
		<span class="if year">
			<span class="year"></span>, 
		</span>
		<span class="author"></span>
		<span class="if url" style="margin-left: 20px">
		  <a class="url" style="color:black; font-size:10px">(view online)</a>
		</span>
		</div>
	</div>
  <div style="">
    <span class="title"></span>
  </div>
</td>
</table>

</div>
<div id="bibtex_display"></div>
