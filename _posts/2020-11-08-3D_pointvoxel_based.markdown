---
layout: post
title:  "Deep Learning on 3D Point Clouds processing - Part V"
date:   2020-11-08
description: "An efficient Deep Learning method joining voxel based and point based approaches : PVCNN"
---

<div style="font-size: 0.8em; text-align: justify;" markdown=1>


Previous Deep Learning models for point cloud processing were essentially voxel based like <a href="#references">[11]</a> or point based methods like <a href="#references">[1]</a>, <a href="#references">[3]</a>, <a href="#references">[6]</a>. Further methods use multi view images or graph representations of point clouds.<br> Voxel-based methods suffer from a memory footprint which grows cubically with the voxel grid resolution <a href="#references">[2]</a>. This prohibits the ability to build architectures supporting high resolution voxel grids as an input. Consequently, the detection of fine grained objects in complex scenes is difficult.<br>
Point based methods are memory-efficient thanks to their sparsity. However, for most of them, 80% of the computation time is spent structuring the point cloud data <a href="#references">[2]</a> : memory access, neighborhood search, etc.<br>
Each of these families of methods have their pros and cons. <a href="#references">[2]</a> proposes a solution that takes the best of each approach to build an optimal and  efficient solution in time and space complexities.<br>
The main contribution in this solution is a  building block (layer) called <b>PVConv</b>. This layer processes the point cloud in its voxel and sparse data representations through two (02) branches. The first branch apply 3D convolutions on a low resolution voxel grid of the data to extract coarse-grained features. The second branch apply MLP layers on the raw point clouds to extract fine grained features missed by the first branch. Let us dive into this model to discover how efficient it is in terms of memory and time consumption while reaching the SOTA benchmark.


## PVCNN : Motivations


Thanks to the regularity of their input, **Voxel-based** models benefit an efficient memory access. Also, the voxel grid resolution allows the application of 3D convolution operations on the grid to learn local patterns. <br>However, most of the cells in a voxel grids are empty. The memory consumption is extensive. As illustrated in <a href="#figure1">Figure 1.a</a>, a **3D U-Net<a href="#references">[11]</a>** model trained on the **part segmentation task** needs about **12 GB of GPU** memory for a voxel grid resolution of **64x64x64** and a batch size of 16. This resolution is relatively low (e.g. if this same resolution were to be applied on a large scene semantic segmentation). **If the resolution is doubled, the GPU need increases to 86 GB**.
<br>
The horizontal axis in <a href="#figure1">Figure 1.a</a> represents the percentage of **distinguishable points** in a point cloud. A distinguishable point occupies a voxel cell alone, in the voxel grid. The higher the resolution of the grid, the higher the percentage of distinguishable points; and the better the point cloud representation in the voxel grid. **It can be seen as a measure of the proper representation of a point cloud in a voxel grid.** An overview of <a href="#figure1">Figure 1.a</a> tells us that a voxel-based method like 3D U-Net needs an astronomous GPU memory to support a well represented point cloud in a voxel grid.
<br>
<center>
<div id="figure1">
  <figure  style="width:100%; margin:0;">
  <img src="{{ '/assets/img/pvcnn_fig1.PNG' | prepend: site.baseurl }}" alt="" style=""> 
  <center  style="font-style: initial;"><b>Figure 1.</b> Illustration of the drawbacks of point/voxel based methods</center>
</figure>
</div>
</center>
<br>
**Point based** methods thanks to the sparsity of the input does not suffer from a high memory footprint. They can even be faster than voxel-based methods.<br>
However, **the irregularity and the unorderness** of the input **increases the memory access time** as illustrated in <a href="#figure1">Figure 1.b</a>. In point based methods, there is generally neighborhood search operations which has a complexity of **$O(n^2)$** or **$O(nlog(n)$** if using a tree representation, where n is the number of representative points in a given layer. In addition to those neighborhood searches, point convolution methods apply additional processing steps called **dynamical kernel operations** (e.g. neighbors permutation before convolution). This is time consuming and represents side operations before the actual computations (e.g. convolutions, etc).<br>
<a href="#figure1">Figure 1.b</a> shows how PVCNN spends less than 10% of its runtime doing side operations unlike the other methods on the same graph.<br>

Point based methods like voxel based methods have their advantages but suffer from the drawbacks we have seen before. In the following lines, we will study a proposal that takes only the most of these approaches.



## PVCNN : Model

<center>
<div id="figure2">
  <figure  style="width:100%; margin:0;">
  <img src="{{ '/assets/img/pvcnn_pvconv.PNG' | prepend: site.baseurl }}" alt="" style=""> 
  <center  style="font-style: initial;"><b>Figure 2.</b> Illustration of the drawbacks of point/voxel based methods</center>
</figure>
</div>
</center>
<br>
The authors used **PointNet<a href="#references">[1]</a>**, **PointNet++<a href="#references">[3]</a>**, **F-PointNet<a href="#references">[12]</a>** as **backbones** in the classification, segmentation and object detection tasks. The major modification done was to replace the shared MLP layers in these architectures by a **PVConv module**, creating **PVCNN** architectures (from PointNet and F-PointNet) and **PVCNN++** from PointNet++. Some skip connections between layers were also used <a href="https://github.com/mit-han-lab/pvcnn">[github]</a>. The core of these new models is PVConv.<br>
PVConv takes as input a set of points with their corresponding features, <a href="#figure2">Figure 2</a>.<br>
* In the **first branch** the point cloud is transformed into a  voxel grid. Then, **3D convolutions** are applied on this voxel grid. To output features per point, a **devoxelization** step follows the convolutions. it corresponds to a **triliniear interpolation** in which we aggregate the features of the nearest voxel cells to each original point in the input. The features are weighted by the distance of their corresponding cell to each point, before the aggregation. In this branch the **resolution of the voxel grid is low**; consequently, this branch specializes in detecting **coarse details**.
* To capture the missed elements due to the low resolution of the voxels, a **second branch** applies **MLPs** on the input points. Those points are mapped into a higher dimensional space, where the local tructure of each point (e.g. relations to neighbors) is encoded through the MLP layers; consequently, this branSch learns the **fine grained local elements**.<br>

The results of these two branches are summed up point-wisely, since they are complementary, yielding an encoding vector for each point. By using a low resolution voxel input, we can apply 3D convolutions to learn coarse local features while maintaining a low memory footprint.  The MLP layers used on raw point clouds do not need dynamical kernel operations and can directly learn subtle elements missed by the 3D convolutions.

## PVCNN : Experiments

The authors evaluated PVCNN on **three (03) tasks** : object part segmentation on ShapeNet<a href="#references">[4]</a>, indoor scene segmentation on S3DIS<a href="#references">[5]</a> and object detection on Kitti Dataset <a href="#references">[8]</a>. PVCNN is built uppon PointNet by replacing shared MLP layers by PVConv modules. The same idea is exploited with PointNet++ to make PVCNN++. For object detection, PVCNN is built from F-PointNet. An efficient version of this model consists in replacing the MLP layers in the instance segmentation network of F-PointNet; a complete version consists in further replacing MLP layers in the box estimation network too.

<center>
<div id="figure3">
  <figure  style="width:100%; margin:0;">
  <img src="{{ '/assets/img/pvcnn_fig3.PNG' | prepend: site.baseurl }}" alt="" style=""> 
  <center  style="font-style: initial;"><b>Figure 3.</b>Comparison of PVCNN with other baselines on ShapeNet Part.<br> Forward pass on GTX 1080 Ti GPU : <b>#batch size=8, #points=2048</b></center>
</figure>
</div>
</center>
<br>
For PVCNN and 3D-UNet<a href="#references">[]</a>, multiple models have been built with different complexities (e.g. by changing the channels dimension) to test the limits in term of runtime and memory consumption. As we can see, PVCNN with the lowest complexity in <a href="#figure3">Figure 3</a> (latency <25ms and memory consumption<1GB) seems to outperform all the point/voxel based methods represented in these graphs except PointCNN, in the part segmentation task. When reaching PointCNN performance, PVCNN remains 2.7 times faster while consuming only $\frac{2}{3}$ of the GPU memory of PointCNN.


<center>
<div id="figure4">
  <figure  style="width:100%; margin:0;">
  <img src="{{ '/assets/img/pvcnn_fig4.PNG' | prepend: site.baseurl }}" alt="" style=""> 
  <center  style="font-style: initial;"><b>Figure 4.</b>Comparison of PVCNN with other baselines on S3DIS.<br> Forward pass on GTX 1080 Ti GPU : #batch size=8, #points=32768</center>
</figure>
</div>
</center>
<br>
In <a href="#figure4">Figure 4</a>, PVCNN++ PVCNN outperforms on the indoor semantic segmentation task. This is not surprising because of the very nature of PVCNN++ which is more adapted to learn local structures in complex scenes.

<center>
<div id="table1">
  <figure  style="width:100%; margin:0;">
  <img src="{{ '/assets/img/pvcnn_table1.PNG' | prepend: site.baseurl }}" alt="" style=""> 
  <center  style="font-style: initial;"><b>Table 1.</b>Result of 3D object detection on KITTI validation dataset.<br>The complete PVCNN outperforms
F-PointNet++ in all categories significantly<br> with 1.5× measured speedup and memory reduction.</center>
</figure>
</div>
</center>
<br>
<a href="#table1">Table 1</a> summarizes the performance of PVCNN compared to its original backbone F-PointNet.

#### Further analysis 

* **Voxel Grid Resolution** : In the first branch of PVConv a relatively low resolution of the voxel grid is acceptable. In additional experiments in part segmentation, the authors proved that by squeezing this resolution again (by half), the latency can be reduced by almost half (**28.9ms vs 50.7ms**) with about only 0.7 mIOU loss (**86.2 to 85.5**).
* In <a href="#figure5">Figure 5</a>, the features learned by both branches are highlighted for each point. The warmer color, represents larger magnitude. The first represent the voxel branch and the second the point branch. The voxel branch learns to capture large parts like a chair seat, the a table surface etc. The point branch, on the other hand, learns more refined parts like  the legs, the wrist of a glass etc. They are indeed complementary.

<center>
<div id="figure5">
  <figure  style="width:100%; margin:0;">
  <img src="{{ '/assets/img/pvcnn_fig5.PNG' | prepend: site.baseurl }}" alt="" style=""> 
  <center  style="font-style: initial;"><b>Figure 5.</b>Representation of the features per point learned in PVCONV branches.</center>
</figure>
</div>
</center>
<br>

### Conclusion
Deep Learning in point cloud processing has brought significance improvements in the different tasks involved. Though, the increasing complexities of the models demands more and more  computation power, limiting the scaling at production level. With PVCNN, the authors proposed a fast and efficient deep learning method for point cloud processing. The module introduced in this paper allows the reduction of memory footprint that grows cubically in voxel based methods, while reducing the memory access and side operations on point clouds (e.g. neighborhood search) in point based methods. Thinking efficient deep learning solutions is crucial, e.g. in real time applications. <a href="#references">[13]</a> introduced the idea of submanifold sparse convolutional networks, where the 3D convolution are aplied on sparse voxel grids. A similar idea is further exploited in <a href="#references">[2]</a>. In the same paper, the authors introduced NAS search <a href="#references">[14]</a> in their method to search an optimal nework architecture under some complexity constraints.


### References
<br>

<textarea id="bibtex_input" style="display:none;">

@misc{liu2019pointvoxel,
      title={Point-Voxel CNN for Efficient 3D Deep Learning}, 
      author={Zhijian Liu and Haotian Tang and Yujun Lin and Song Han},
      year={2019},
      eprint={1907.03739},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      pos={2}
}


@INPROCEEDINGS{8099499,
author={R. Q. {Charles} and H. {Su} and M. {Kaichun} and L. J. {Guibas}},
booktitle={2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}, title={PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation},
year={2017},
volume={},
number={},
pages={77-85},
doi={10.1109/CVPR.2017.16},
pos={1}}
}

@inproceedings{qi2017pointnet++,
  title={Pointnet++: Deep hierarchical feature learning on point sets in a metric space},
  author={Qi, Charles Ruizhongtai and Yi, Li and Su, Hao and Guibas, Leonidas J},
  booktitle={Advances in neural information processing systems},
  pages={5099--5108},
  year={2017},
  pos={3}
}




@article{shapenet,
author = {Yi, Li and Kim, Vladimir G. and Ceylan, Duygu and Shen, I-Chao and Yan, Mengyan and Su, Hao and Lu, Cewu and Huang, Qixing and Sheffer, Alla and Guibas, Leonidas},
title = {A Scalable Active Framework for Region Annotation in 3D Shape Collections},
year = {2016},
issue_date = {November 2016},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {35},
number = {6},
issn = {0730-0301},
url = {https://doi.org/10.1145/2980179.2980238},
doi = {10.1145/2980179.2980238},
journal = {ACM Trans. Graph.},
month = nov,
articleno = {210},
numpages = {12},
keywords = {shape analysis, active learning},
pos={4}
}

@INPROCEEDINGS{s3dis,  author={I. {Armeni} and O. {Sener} and A. R. {Zamir} and H. {Jiang} and I. {Brilakis} and M. {Fischer} and S. {Savarese}},  booktitle={2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},   title={3D Semantic Parsing of Large-Scale Indoor Spaces},   year={2016},  volume={},  number={},  pages={1534-1543},  doi={10.1109/CVPR.2016.170},
pos={5}}

@misc{pointcnn,
      title={PointCNN: Convolution On X-Transformed Points}, 
      author={Yangyan Li and Rui Bu and Mingchao Sun and Wei Wu and Xinhan Di and Baoquan Chen},
      year={2018},
      eprint={1801.07791},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      pos={6}
}


@misc{spidercnn,
      title={SpiderCNN: Deep Learning on Point Sets with Parameterized Convolutional Filters}, 
      author={Yifan Xu and Tianqi Fan and Mingye Xu and Long Zeng and Yu Qiao},
      year={2018},
      eprint={1803.11527},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      pos={7}
}

@article{kitti,
author = {A Geiger and P Lenz and C Stiller and R Urtasun},
title ={Vision meets robotics: The KITTI dataset},
journal = {The International Journal of Robotics Research},
volume = {32},
number = {11},
pages = {1231-1237},
year = {2013},
doi = {10.1177/0278364913491297},
URL = {https://doi.org/10.1177/0278364913491297},
eprint = {https://doi.org/10.1177/0278364913491297},
abstract = {},
pos={8}
}

@INPROCEEDINGS{rsnet,
  author={Q. {Huang} and W. {Wang} and U. {Neumann}},
  booktitle={2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition}, 
  title={Recurrent Slice Networks for 3D Segmentation of Point Clouds}, 
  year={2018},
  volume={},
  number={},
  pages={2626-2635},
  doi={10.1109/CVPR.2018.00278},
  pos={9}
}

@misc{dgcnn,
      title={Dynamic Graph CNN for Learning on Point Clouds}, 
      author={Yue Wang and Yongbin Sun and Ziwei Liu and Sanjay E. Sarma and Michael M. Bronstein and Justin M. Solomon},
      year={2019},
      eprint={1801.07829},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      pos={10}
}

@misc{3d-unet,
      title={3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation}, 
      author={Özgün Çiçek and Ahmed Abdulkadir and Soeren S. Lienkamp and Thomas Brox and Olaf Ronneberger},
      year={2016},
      eprint={1606.06650},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      pos={11}
}

@misc{qi2018frustum,
title={Frustum PointNets for 3D Object Detection from RGB-D Data}, 
author={Charles R. Qi and Wei Liu and Chenxia Wu and Hao Su and Leonidas J. Guibas},
year={2018},
eprint={1711.08488},
archivePrefix={arXiv},
primaryClass={cs.CV},
pos={12}
}

@misc{graham20173d,
      title={3D Semantic Segmentation with Submanifold Sparse Convolutional Networks}, 
      author={Benjamin Graham and Martin Engelcke and Laurens van der Maaten},
      year={2017},
      eprint={1711.10275},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      pos={13}
}

@misc{tang2020searching,
      title={Searching Efficient 3D Architectures with Sparse Point-Voxel Convolution}, 
      author={Haotian Tang and Zhijian Liu and Shengyu Zhao and Yujun Lin and Ji Lin and Hanrui Wang and Song Han},
      year={2020},
      eprint={2007.16100},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      pos={14}
}

</textarea>

<div class="bibtex_template" style="">
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
<p id="bibtex_display"></p>


</div>
