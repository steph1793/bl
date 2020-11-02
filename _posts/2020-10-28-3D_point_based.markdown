---
layout: post
title:  "Point-Based Deep Learning methods for 3D point cloud processing"
date:   2020-10-28
description: "Add a description"
---

<p style="text-align: justify; font-size: 0.9em"><span class="dropcap">T</span>he development of 3D data acquisition technologies like LIDAR or RGB-D cameras, the increasing computing capacity and the recent breakthroughs in Deep Learning has favoured the use of 3D data in a variety of fields for perception : robotics, medical imaging, self-driving cars etc. Those data can be represented as point clouds, RGB-D images or meshes. Despite 2D images, 3D data provide a depth information.
<br>In robotics and self-driving cars, perception is crucial. A robot necessarily needs to localize the wrist of a glass to be able to grab it; an autonomous vehicle needs to detect the other vehicles, the traffic lights, the road, to be able to drive. This necessity has given rise to a number of tasks in 3D data processing like object detection, semantic segmentation, instance segmentation for scene understanding, object part segmentation, object classification <b>[1]</b>.
</p>

# PointNet

<p style="text-align: justify; font-size: 0.9em">PointNet is an architecture that directly consumes as input a raw point cloud represented by a matrix $M \in \mathbb{R}^{N * m}$ where N is the number of points and m the dimensionality of the features (xyz coordinates + rgb color + normal + intensity etc). Note that $m=3$ if the model only consumes xyz coordinates.<br>
To be able to process raw point clouds the neural network must overcome some challenges due to the very nature of the input.</p>
* <p style="text-align: justify; font-size: 0.9em"><b>Unorderness</b> : there is no consistent natural canonical order for our set of points. Therefore for the same objects we have $N!$ possible permutations of our matrix. The network must be insensitive to those permutations.</p>
* <p style="text-align: justify; font-size: 0.9em"><b>Irregularity</b> : Despite 2D images, there is no regular structure in a point cloud. The points are irregularly disposed in the 3D space, sparser in some areas and denser in others, with interactions among them and a notion of neighborhood. The model needs to learn those local interactions since they are the ones that define a shape.</p>
* <p style="text-align: justify; font-size: 0.9em"><b>Sensitivity to transformations</b> : when being applied a rigid transformation (or non rigid for organic objects <b>[pointNet++]</b>), the point cloud yields a completly different matrix, that still represents the same object. The model must be insentive to those transformations.</p>

### PointNet : Model

The model proposed in [Pointnet] is built uppon three (03) key modules:
* A symmetric function invariant to the points order $f \mapsto f({x_1, x_2, ..., x_n}) = g(h(x_1), h(x_2), ..., h(x_n))$ where $h$ is MLP layers that extract features frm each point by mapping them into a higher K-dimensional space; $g : \mathbb{R}^{N*K} \rightarrow \mathbb{R}^K$ is a global max pooling layer accross the N points that will output an encoding representing the entire point cloud.<br>

* Local and global information Aggregation : the encoding vector built with the symmetric function can be further processed for classification as illustrated in [figure 1]; or, this feature vector can be reconcatenated to the features of each point, which will be further processed through MLP layers for segmentation. In this case, each point is represented by a more local feature information extracted by the symmetric function and a global information regarding the entire pointcloud provided by the max pooling layer. Roughly speaking, the global information informs about the shape of the point cloud and the local information about the local structure for each point.

* Joint alignment Network : this module helps the model to be invariant to some geometric transformations (e.g. rigid transformations) of the point clouds. For this, a sub-network called T-Net similar to the entire PointNet architecture is created to learn a transformation matrix from the point cloud [figure model]. This matrix is applied on the point cloud to re-align it before the subsequent steps. The same idea can be applied on the features of the point cloud. In this case, to ease the optimization, a regularization term is added : $ L_{reg} = \|\|{I - AA^T}\|\|^2_F $ where $A$ is the predicted transformation matrix, $\|\|.\|\|_F$ is the Frobenius norm. This regularization constraint the predicted matrix to be a rotation matrix.


The overall architecture can be formulated as :


### PointNet : Why does it work

<div class="theorem"><b style="font-style: initial;">1.</b>
Suppose $f : \mathcal{X} \rightarrow \mathbb{R}$ is a continuous function w.r.t Hausdorff distance $d_H(.,.)$. $\forall \epsilon > 0$, $\exists$ a continuous function $h$ and a symmetric function $g(x_1, ..., x_n) = \gamma \circ MAX$, such that for any $S \in \mathcal{X}$,
\begin{align*}
| f(S) - \gamma(MAX_{x_i \in S} \{h(x_i)\}) | < \epsilon
\end{align*}

where $x_1,...,x_n$ is the full list of elements in S ordered arbitrarily, $\gamma$ a continuous function, and $MAX$ is a vector max operator that takes n vectors as input and returns a new vector of the element-wise maximum.
</div>
Here $h$ is in fact a function, that maps a point into a single channel of the point feature vector. $h$ represents the feed-forward network before the max-pooling and $\gamma$ represents the subsequent layers after the max-pooling. Be aware, the formulation here is channel-wise : $h$ does not output an entire feature vector bu only a channel of that feature vector. The entire feature vector can be obtained by concatenating the results of all the $h$'s.


This theorem states that PointNet can approximate any continuous function of set of points to some extent ($\epsilon$ precision); if given sufficiennt neurons (universal approximation theorem). Therefore, under a specific supervision, the network can learn to approximate the continuous function of set of points for classification or segmentation.



<div class="theorem"><b style="font-style: initial;">2.</b>
Suppose $u : \mathcal{X} \rightarrow \mathbb{R}^K$ such that $u=MAX\{h(x_i)\}$ and $f = \gamma \circ u$. Then, <br>
(a) $\forall S$, $\exists \mathcal{C}_S, \mathcal{N}_S \subseteq \mathcal{X}$, $f(T) = f(S)$ if $\mathcal{C}_S \subseteq T \subseteq \mathcal{N}_S$;<br>
(b) $|\mathcal{C}_S| \leq K$
</div>

In this setting, $h$ outputs features vector per point. The max-pooling layer is channel-wise accross the points. $K$ represents the dimensionality of the last layers of $h$ and per extension the dimentionality of the global feature after the max-pooling layer.

This theorem states that PointNet is a robust neural network. It is robust to some extra noise (e.g. outlier points) added to the point cloud. It also states that even if some points are removed from the point cloud, as far some particular points called critial points ($\mathcal{C}_S$) are kept, the network will be robust to this information loss.<br>
The critical points set cardinality is bounded by $K$. The max pooling layer outputs a global encoding which is formed by the maximum values coming from $K$ points among the $N$ points. As proved in the experiments, those $K$ points form the critical point set.



### PointNet : Experiments

#### Experiments on Classification Task
For this task, a PointNet was trained and evaluated on ModelNet40 dataset [28]. This dataset is made of 12,311 CAD objects. 1024 points were sampled from the surface of those objects.


<center>
<table style="font-size: 0.6em; width: 75%; align-self: center;" >
<thead>
<tr>
<th>Models</th>
<th>input</th>
<th>#views</th>
<th>accuracy<br> avg./class</th>
<th>Overall<br> accuracy</th>
<th>#params</th>
<th>FLOPs/<br>sample</th>
</tr>
</thead>
<tbody>
<tr>
<td>SPH [ ]</td>
<td>mesh</td>
<td>-</td>
<td>66.2</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>3DShapeNets [ ]<br />VoxNet [ ]<br /><b>SubVolume [ ]</b></td>
<td>Voxel Grid<br />Voxel Grid<br />Voxel Grid</td>
<td>1<br />12<br />20</td>
<td>77.3<br />83.0<br />86.0</td>
<td>84.7<br />85.9<br /><b>89.2</b></td>
<td>-<br>-<br>16.6M</td>
<td>-<br>-<br>3633M</td>
</tr>
<tr>
<td>LFD [ ]<br /><b>MVCNN [ ]</b></td>
<td>images<br>images</td>
<td>10<br />80</td>
<td>75.5<br /><b>90.1</b></td>
<td>-<br />-</td>
<td>-<br />60.0M</td>
<td>-<br />62057</td>
</tr>
<tr>
<td>Baseline [ ]<br /><b>PointNet [ ]</b></td>
<td>point cloud<br>point cloud</td>
<td>-<br/>1</td>
<td>72.6<br/>86.2</td>
<td>77.4<br/><b>89.2</b></td>
<td>-<br />3.5M</td>
<td>-<br />440M</td>
</tr>
</tbody>
<caption style="font-style: initial;">Table 1 : Classification results on ModelNet40</caption>
</table>
</center>

Table 1 summarizes the results of PointNet and the benchmark models at that time. The baseline model is just a MLP network trained on some hand-crafted features extracted on each point cloud. While PointNet reaches the performances of [subvolume ref], with a small with [mvcnn ref] it is much more efficient in terms of memory consumption (number of parameters) and runtime complexity. Also, [subvolume] and [mvcnn] takes as inout multiple rotations or views of the point clouds, whereas PointNet performs in a single shot.


#### Experiments on Part Segmentation Task
In this experiment, PointNet was evaluated on its capability to identify fine-grained details and segment a point cloud in its different parts.
The dataset used was ShapeNet [ref]. It contains 16,881 shapes from 16 categories, with a total of 50 object parts.

<center>
<div style=" display: table; width: 100%">
  <figure  style="width:90%; margin:0;">
  <img src="{{ '/assets/img/pointnetSeg.PNG' | prepend: site.baseurl }}" alt="" style=""> 
  <center style="font-style: initial;">Fig1. - This is an example figcaption</center>
</figure>
<div markdown="1" style="font-size: 0.6em; width: 30%; align-self: center;  vertical-align: middle; display:table-cell;">

|**Models**|Yi []| 3DCNN []| **PointNet** |
|**mIOU**|81.4|79.4| **83.7** |

<center style="font-style: initial;">Table 2 : Part Segmentation Classification on ShapeNet.</center>
</div>
</div>
</center>
<br><br>
In table 2, 3DCNN is a voxel based 3D convolutional method created by the authors as a baseline for this task. As illustrated in Fig.1, PointNet can segment realistic partial point clouds.



#### Experiments on Semantic Segmentation
The experiment is done on the Stanford 3D dataset which is made of indoor scenes : 6 areas, 271 rooms.

### References
<br>

<textarea id="bibtex_input" style="display:none;">
@misc{guo2020deep,
      title={Deep Learning for 3D Point Clouds: A Survey}, 
      author={Yulan Guo and Hanyun Wang and Qingyong Hu and Hao Liu and Li Liu and Mohammed Bennamoun},
      year={2020},
      eprint={1912.12033},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      pos={1}
}
}

@inproceedings{su2015multi,
  title={Multi-view convolutional neural networks for 3d shape recognition},
  author={Su, Hang and Maji, Subhransu and Kalogerakis, Evangelos and Learned-Miller, Erik},
  booktitle={Proceedings of the IEEE international conference on computer vision},
  pages={945--953},
  year={2015},
  pos={2}
}
@article{leng20153d,
  title={3D object retrieval with stacked local convolutional autoencoder},
  author={Leng, Biao and Guo, Shuang and Zhang, Xiangyang and Xiong, Zhang},
  journal={Signal Processing},
  volume={112},
  pages={119--128},
  year={2015},
  publisher={Elsevier},
  pos={3}
}

@inproceedings{bai2016gift,
  title={Gift: A real-time and scalable 3d shape search engine},
  author={Bai, Song and Bai, Xiang and Zhou, Zhichao and Zhang, Zhaoxiang and Jan Latecki, Longin},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={5023--5032},
  year={2016},
  pos={4}
}

@article{krizhevsky2017imagenet,
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

@INPROCEEDINGS{7353481,
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

@INPROCEEDINGS{8578570,
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

@article{wang2019dynamic,
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

@INPROCEEDINGS{8099499,
author={R. Q. {Charles} and H. {Su} and M. {Kaichun} and L. J. {Guibas}},
booktitle={2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}, title={PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation},
year={2017},
volume={},
number={},
pages={77-85},
doi={10.1109/CVPR.2017.16},
pos={11}}
}

@inproceedings{qi2017pointnet++,
  title={Pointnet++: Deep hierarchical feature learning on point sets in a metric space},
  author={Qi, Charles Ruizhongtai and Yi, Li and Su, Hao and Guibas, Leonidas J},
  booktitle={Advances in neural information processing systems},
  pages={5099--5108},
  year={2017},
  pos={12}
}

@INPROCEEDINGS{9010002,  author={H. {Thomas} and C. R. {Qi} and J. {Deschaud} and B. {Marcotegui} and F. {Goulette} and L. {Guibas}},  booktitle={2019 IEEE/CVF International Conference on Computer Vision (ICCV)},   title={KPConv: Flexible and Deformable Convolution for Point Clouds},   year={2019},  volume={},  number={},  pages={6410-6419},  doi={10.1109/ICCV.2019.00651}, pos={13}}

@incollection{NIPS2018_7362,
title = {PointCNN: Convolution On X-Transformed Points},
author = {Li, Yangyan and Bu, Rui and Sun, Mingchao and Wu, Wei and Di, Xinhan and Chen, Baoquan},
booktitle = {Advances in Neural Information Processing Systems 31},
editor = {S. Bengio and H. Wallach and H. Larochelle and K. Grauman and N. Cesa-Bianchi and R. Garnett},
pages = {820--830},
year = {2018},
publisher = {Curran Associates, Inc.},
pos={14}
}

@inproceedings{Liu2019PointVoxelCF,
  title={Point-Voxel CNN for Efficient 3D Deep Learning},
  author={Zhijian Liu and Haotian Tang and Yujun Lin and Song Han},
  booktitle={NeurIPS},
  year={2019},
  pos={15}
}

@inproceedings{jaritz2019multi,
  title={Multi-view pointnet for 3d scene understanding},
  author={Jaritz, Maximilian and Gu, Jiayuan and Su, Hao},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision Workshops},
  pages={0--0},
  year={2019},
  pos={16}
}

@misc{choy20194d,
      title={4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks}, 
      author={Christopher Choy and JunYoung Gwak and Silvio Savarese},
      year={2019},
      eprint={1904.08755},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      pos={17}
}

 @inproceedings{tang2020searching,
    title     = {Searching Efficient 3D Architectures with Sparse Point-Voxel Convolution},
    author    = {Tang, Haotian* and Liu, Zhijian* and Zhao, Shengyu and Lin, Yujun and Lin, Ji and Wang, Hanrui and Han, Song},
    booktitle = {European Conference on Computer Vision},
    year      = {2020},
    pos={18}
 } 

 @article{mdpi1111,
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
