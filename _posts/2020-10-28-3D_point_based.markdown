---
layout: post
title:  "Point-Based Deep Learning methods for 3D point cloud processing"
date:   2020-10-28
description: "Add a description"
---

<div style="font-size: 0.8em; text-align: justify;" markdown=1>

In this article, we are going to discuss two (02) point based Deep Learning methods : **PointNet <a href="#references">[1]</a>** and **PointNet++ <a href="#references">[2]</a>**. PointNet is an important architecture for point cloud processing: it is the first deep learning method that directly processes raw sparse point clouds. The authors provide some theoretical insights to support their proposal. PointNet++ is based on PointNet. Thanks to its hierarchical architecture, it is more adapted to process complex scenes with numerous local structures.


## PointNet


<center>
<div id="figure1">
  <figure  style="width:80%; margin:0;">
  <img src="{{ '/assets/img/pointnet.PNG' | prepend: site.baseurl }}" alt="" style=""> 
  <center  style="font-style: initial;"><b>Figure 1.</b> PointNet architecture</center>
</figure>
</div>
</center>
<br>
<p>PointNet is an architecture that directly consumes a raw point cloud represented by a matrix $M \in \mathbb{R}^{N * m}$ where N is the number of points and m the dimensionality of the features (xyz coordinates + rgb color + normal + intensity etc). Note that $m=3$ if the model only consumes xyz coordinates.<br>
To be able to process raw point clouds the neural network must overcome some challenges due to the very nature of the input.</p>
* <b>Unorderness</b> : there is no consistent natural canonical order for a set of 3D points. Therefore for the same objects we have $N!$ possible permutations of our matrix. The network must be insensitive to those permutations.
* <b>Irregularity</b> : Despite 2D images, there is no regular structure in a point cloud. The points are irregularly disposed in the 3D space, sparser in some areas and denser in others, with interactions among them and a notion of neighborhood. The model needs to learn those local interactions since they are the ones that define a shape.
* <b>Sensitivity to transformations</b> : when being applied a rigid transformation (or non rigid for organic objects <b><a href="#references">[2]</a></b>), the point cloud yields a completly different matrix, that still represents the same object. The model must be insentive to those transformations.

### PointNet : Model

The model proposed in **<a href="#references">[1]</a>** is built uppon three (03) key modules, as illustrated in <a href="#figure1">Figure 1</a>:
* <b>A symmetric function</b> invariant to the points order $f \mapsto f({x_1, x_2, ..., x_n}) = g(h(x_1), h(x_2), ..., h(x_n))$ where $h$ is MLP layers that extract features from each point by mapping them into a higher K-dimensional space; $g : \mathbb{R}^{N*K} \rightarrow \mathbb{R}^K$ is a global max pooling layer accross the N points that will output an encoding representing the entire point cloud.<br><br>
* <b>Local and global information Aggregation</b> : the encoding vector built with the symmetric function can be further processed for classification as illustrated in <a href="#figure1">Figure 1</a>; or, this feature vector can be concatenated to the features of each point, which will be further processed through MLP layers for segmentation. In this case, each point is represented by a more local feature information extracted by the shared MLP layers (in the symmetric function) and a global information regarding the entire pointcloud provided by the max pooling layer. Roughly speaking, the global information informs about the shape of the point cloud and the local information about the local structure for each point.<br><br>
* <b>Joint alignment Network</b> : this module helps the model to be invariant to some geometric transformations (e.g. rigid transformations) of the point clouds. For this, a sub-network called T-Net similar to the entire PointNet architecture is created to learn a transformation matrix from the point cloud. This matrix is applied on the point cloud to re-align it before the subsequent steps. The same idea can be applied on the features of the point cloud. In this case, to ease the optimization, a regularization term is added : $ L_{reg} = \|\|{I - AA^T}\|\|^2_F $ where $A$ is the predicted transformation matrix, $\|\|.\|\|_F$ is the Frobenius norm. This regularization constraint the predicted matrix to be a rotation matrix.


### PointNet : Why does it work

<p class="theorem" ><b style="font-style: initial;">1.</b>
Suppose $f : \mathcal{X} \rightarrow \mathbb{R}$ is a continuous function w.r.t Hausdorff distance $d_H(.,.)$. $\forall \epsilon > 0$, $\exists$ a continuous function $h$ and a symmetric function $g(x_1, ..., x_n) = \gamma \circ MAX$, such that for any $S \in \mathcal{X}$,
\begin{align*}
| f(S) - \gamma(MAX_{x_i \in S} \{h(x_i)\}) | < \epsilon
\end{align*}

where $x_1,...,x_n$ is the full list of elements in S ordered arbitrarily, $\gamma$ a continuous function, and $MAX$ is a vector max operator that takes n vectors as input and returns a new vector of the element-wise maximum.
</p>
Here $h$ is a function, that maps a point into a single channel of the point feature vector. $h$ represents the feed-forward network before the max-pooling and $\gamma$ represents the subsequent layers after the max-pooling. Be aware, the formulation here is channel-wise : $h$ does not output an entire feature vector but only a channel of that feature vector. The entire feature vector can be obtained by concatenating the results of all the $h$'s.


This theorem states that **PointNet ( $\gamma(MAX_{x_i \in S} \{h(x_i)\})$ ) can approximate any continuous function of set of points to some extent ($\epsilon$ precision)**; if given sufficient neurons (universal approximation theorem). Therefore, under a specific supervision, the network can learn to approximate the continuous function of set of points that predicts a classification or a segmentation.



<p class="theorem"><b style="font-style: initial;">2.</b>
Suppose $u : \mathcal{X} \rightarrow \mathbb{R}^K$ such that $u=MAX\{h(x_i)\}$ and $f = \gamma \circ u$. Then, <br>
(a) $\forall S$, $\exists \mathcal{C}_S, \mathcal{N}_S \subseteq \mathcal{X}$, $f(T) = f(S)$ if $\mathcal{C}_S \subseteq T \subseteq \mathcal{N}_S$;<br>
(b) $|\mathcal{C}_S| \leq K$
</p>

In this setting, $h$ outputs features vector per point. The max-pooling layer is channel-wise accross the points. $K$ represents the dimensionality of the last layers of $h$ and per extension the dimentionality of the global feature after the max-pooling layer.

**This theorem states that PointNet is a robust neural network.** It is robust to some extra noise (e.g. outlier points) added to the point cloud. It also states that even if some points are removed from the point cloud, as far some particular points called critial points ($\mathcal{C}_S$) are kept, the network will be robust to this information loss.<br>
The critical points set cardinality is bounded by $K$. The max pooling layer outputs a global encoding which is formed by the maximum values coming from $K$ points among the $N$ points. As proved in the experiments, those $K$ points form the critical point set.



### PointNet : Experiments

#### Experiments on Classification Task
For this task, a PointNet model was trained and evaluated on **ModelNet40 dataset <a href="#references">[3]</a>**. This dataset is made of 12,311 CAD objects. 1024 points were sampled from the surface of those objects.


<center>
<table style="font-size: 0.6em; width: 75%; align-self: center;"  id="table1">
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
<td>SPH <a href="#references">[4]</a></td>
<td>mesh</td>
<td>-</td>
<td>66.2</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>3DShapeNets <a href="#references">[3]</a><br />VoxNet <a href="#references">[5]</a><br /><b>SubVolume <a href="#references">[6]</a></b></td>
<td>Voxel Grid<br />Voxel Grid<br />Voxel Grid</td>
<td>1<br />12<br />20</td>
<td>77.3<br />83.0<br />86.0</td>
<td>84.7<br />85.9<br /><b>89.2</b></td>
<td>-<br>-<br>16.6M</td>
<td>-<br>-<br>3633M</td>
</tr>
<tr>
<td>LFD <a href="#references">[3]</a><br /><b>MVCNN <a href="#references">[7]</a></b></td>
<td>images<br>images</td>
<td>10<br />80</td>
<td>75.5<br /><b>90.1</b></td>
<td>-<br />-</td>
<td>-<br />60.0M</td>
<td>-<br />62057</td>
</tr>
<tr>
<td>Baseline <a href="#references">[1]</a><br /><b>PointNet <a href="#references">[1]</a></b></td>
<td>point cloud<br>point cloud</td>
<td>-<br/>1</td>
<td>72.6<br/>86.2</td>
<td>77.4<br/><b>89.2</b></td>
<td>-<br />3.5M</td>
<td>-<br />440M</td>
</tr>
</tbody>

</table>
<center style="font-style: initial;"><b>Table 1</b> : Classification results on ModelNet40</center>
</center>
<br>
<a href="#table1">Table 1</a> summarizes the results of PointNet and the benchmark models at that time. The baseline model is a MLP network trained on some hand-crafted features extracted on each point cloud. While PointNet reaches the performances of **Subvolume <a href="#references">[6]</a>**, with a gap with **MVCNN <a href="#references">[7]</a>**, it is much more efficient in terms of memory consumption (number of parameters) and runtime complexity. Also, **Subvolume** and **MVCNN** takes as input multiple rotations or views of the point clouds, whereas PointNet performs in a single shot.


#### Experiments on Part Segmentation Task
In this experiment, PointNet was evaluated on its capability to identify fine-grained details and segment a point cloud in its different parts.
The dataset used was **ShapeNet <a href="#references">[8]</a>**. It contains 16,881 shapes from 16 categories, with a total of 50 object parts.

<center>
<div style=" display: table; width: 80%" id="figure2">
  <div  style="width:70%; margin:0; vertical-align: middle; display:table-cell;">
  <img src="{{ '/assets/img/pointnetSeg.PNG' | prepend: site.baseurl }}" alt="" style=""> 
  <center style="font-size: 0.8em;"><b>Figure 2</b>. PointNet qualitative results</center>
</div>
<div markdown="1" style="font-size: 0.6em; width:30%; vertical-align: middle; display:table-cell;" id="table2">

|**Models**|Yi <a href="#references">[8]</a>| 3DCNN <a href="#references">[13]</a>| **PointNet** |
|**mIOU**|81.4|79.4| **83.7** |

<center ><b>Table 2</b> : Part Segmentation Classification on ShapeNet.</center>
</div>
</div>
</center>
<br><br>
In <a href="">Table 2</a>, 3DCNN is a voxel based 3D convolutional method created by the authors as a baseline for this task. As illustrated in <a href="#figure2">Figure 2</a>, PointNet can segment realistic partial point clouds.



#### Experiments on Semantic Segmentation
The experiment is done on the **Stanford 3D dataset <a href="#references">[9]</a>** which is made of indoor scenes : 6 areas, 271 rooms. PointNet was compared to a baseline model made of MLP layers on hand-crafted features for each point. PointNet outperformed this baseline by **+27 points (47.71% vs 20.12%)** on the **mIOU** metric. The scores remain relatively low however.

<center>
<div>
  <figure  style="width:80%; margin:0;">
  <img src="{{ '/assets/img/pointNetSemSeg.PNG' | prepend: site.baseurl }}" alt="" style=""> 
  <center style="font-style: initial;"><b>Figure 3</b>. PointNet Semantic Segmentation Qualitative Results</center>
</figure>
</div>
</center>


#### Further Analysis
* To tackle the unorderness issue related to point clouds, other approaches have been proposed. For example, one could treat the point clouds as sequences and augment the data with all the permutations possible of that same point cloud. The network  would then be a reccurent neural network (**LSTM <a href="#references">[10]</a>**). However, as shown in **<a href="#references">[1]</a>** and **<a href="#references">[11]</a>**, this configuration is not insensitive to the points order. Thus, the performance is not better.<br><br>
* T-Net is a subnetwork designed in PointNet for the alignement of point clouds, to be robust to some initial transformations. When applied on the input and the features with the regularization term $L_{reg}$, it outperforms PointNet Vanilla (without T-Net) by **$\simeq 2$ pts (89.2% vs 87.1%)** in a classification task. This somehow, proves the usefulness of the alignement network<br><br>
* <b>Robustness</b> : In  the theoretical analysis, it has been proved that as far as set of critical points are kept in the point cloud, PointNet is robust to the points set reduction. This result has been empirically confirmed, as illustrated in <a href="#figure4">Figure 4</a>.

<div id="figure4">
  <figure  style="width:100%; margin:0;">
  <img src="{{ '/assets/img/PointnetRobust.PNG' | prepend: site.baseurl }}" alt="" style=""> 
  
</figure>
<p style="font-size: 0.8em; text-align: justify; padding-left: 50px"><b>Figure 4</b>. Robustness experiments in the classification task. The figure on the left illustrates the robustness of PointNet to points reduction. The blue curve represents PointNet models trained on point sampled on shapes surfaces using a <b>Furthest Sampling algorithm</b>. The curve in the middle illustrates the robustness of PointNet to outliers points added to the original point clouds. The figure on the left shows that by perturbating the points at some extent (0.05 std), the performance is not drastically impacted.</p>
</div>


* <b>Critical point sets and Upper Bounds Shapes</b> : A Critical  point set is the subset of the point cloud which features contribute in the global encoding of the point cloud. They can be seen as the skeletton of the point cloud necessary to the good prediction of PointNet. <a href="#figure5">Figure 5</a> represent some critical sets and upper bound shapes of some objects on which PointNet remains robust; even when the shape seems different from the original point cloud.

<center>
  <figure  style="width:50%;" id="figure5">
  <img src="{{ '/assets/img/pointNetRobust2.PNG' | prepend: site.baseurl }}" alt="" style=""> 
  <center style="font-style: initial; text-align: justify;"><b>Figure 5</b>. Critical Point sets and upper bounds shapes for some examples, along with their PointNet segmentations.</center>
</figure>
</center>


* <b>Failure Analysis</b> : Despite the breakthrough coming with PointNet, this method suffers some issues especially in the segmentation task. PointNet like a lot of other segmentation methods struggles to correctly segment points at boundaries. Also, for rare categories or exotic shapes, PointNet hardly generalizes to other objects of the same category. This points out the weakness of PointNet in non seen Out-Of distribution objects. Finally, in the semantic segmentation, PointNet shows some difficulties to generalize on complex scenes and capture fine-grained local structures. The relatively low scores on the semantic segmentation task seem to confirm this observation.




## PointNet++
<center>
<div id="figure6">
  <figure  style="width:100%; margin:0;">
  <img src="{{ '/assets/img/PointNet++.PNG' | prepend: site.baseurl }}" alt="" style=""> 
  <center style="font-style: initial;"><b>Figure 6</b>. PointNet++ architecture</center>
</figure>
</div>
</center>
<br>
PointNet has been a breakthrough for Deep Learning on 3D point clouds. By design, it overcomes the issues related to those data like the unorderness, the sparsity and the irregularity. This method reached the state of the art benchmark in 2017, with a considerable reduction of the time and space complexities compared to the benchmark. However, PointNet is not  suitable to capture local structures in complex scenes in semantic segmentation, even if we have some sucess cases as illustrated in <a href="#figure3">Figure 3</a>.
<br>
The solution proposed in **PointNet++ <a href="#references">[2]</a>** is to apply hierarchically, a PointNet architecture on partitionned local neighborhoods in the point cloud. More specifically, PointNet++ is based on these three (03) ideas:
* the partitionning of the point cloud into neighborhoods which are further processed one by one to efficiently learn local structures in a complex scene.
* a convolutional setting, where a local feature learner (PointNet) is applied to each partitioned neighborhood to produce an aggregated feature vector. 
* A hierarchical setting that enables the model to build an encoding vector (or a small set of encoding vectors) representing the entire point cloud.


### PointNet++ : The Model


PointNet++ is made of 2 parts as illustrated in <a href="#figure6">Figure 6</a>. The first part is an **Encoder**. Regarding the task to perform, the second part is either a **Decoder** for segmentation task or a **MLP classifier** for classification.
<br>
The encoder is made of identical successive layers called **Set Abstraction (SA)**. A Set Abstraction layer folds as follow :
* **A Sampling layer** : where a subset of points called representative points are sampled from the Point cloud using a **'Furthest Point Sampling' algorithm (FPS)**. FPS sampling  allows a better coverage than a random sampling.
* **A Grouping layer** : For each representative point, a neighborhood is built using a Ball Query or a K-Nearest-Neighbors algorithm. The authors during their experiments proved that Ball queries work better.
* **The PointNet layer** : A PointNet architecture is sequentially applied to each neighborhood, in a convolutional style to output an encoding vector.
At the end of a set abstraction layer, we get a smaller set of representative points, each with a richer feature vector. This subset is passed to the next SA layer, and so on and so forth, until we get a single feature vector  or a smaller set of feature vectors encoding the entire point cloud.
<br>

**The Decoder** (segmentation task):
<br>
The decoder propagates the output of the encoder to the original point cloud, <a href="#figure6"> Figure 6</a>. In this process, the model builds richer features made of a local and global information for each point. This is achieved through identical successive layers called **Feature Propagation (FP) layers**. A SA layer takes as input a set of representative points (or original point cloud) $P_a$ with some features for each point and outputs a subset $P_b$. The corresponding FP layer will interpolate back the features of $P_b$ to $P_a$. The interpolation is a weighted sum of the features of the neighbors from $P_b$ to each point from $P_a$.

\begin{align}
f(x_{P_a}^i) = \frac{\sum^k_{j=1}w(x_{P_a}^i, x_{P_b}^j)f(x_{P_b}^j)}{\sum^k_{j=1}w(x_{P_a}^i, x_{P_b}^j)}
\end{align}
where $x_{P_a}^i$ is the $i-th$ point in $P_a$, $f(x_{P_a}^i)$ is the feature vector of that point, and $w(x_{P_a}^i, x_{P_b}^j)$ is the inverse of the distance between $x_{P_a}^i$ and $x_{P_b}^j$. More weights are attributed to closer neighbors in the interpolation.
<br>
In a FP layer, after the interpolation a skip connection is made with the corresponding SA layer through point-wise features concatenation. A 'unit-pointnet' similar to a unit size kernel convolution is further applied to aggregate the concatenation result for each point.

### PointNet++ : Improvements
<center>
<div id="figure7">
  <figure  style="width:40%; margin:0;">
  <img src="{{ '/assets/img/MSG-MRG.PNG' | prepend: site.baseurl }}" alt="" style=""> 
</figure>
</div>

  <p style="font-style: initial;"><b>Figure 7</b>. Figure a : Multi Scale Grouping in a given neighborhood. Each cone represents a PointNet architecture. Figure b : Multi Resolution Grouping with 2 PointNets represented by the types of cones.</p>
</center>


PointNet++ is sensitive to the ball query radius used in the Grouping layer.  A small radius in a very sparse zone of a point cloud will yield a poor neighborhood while a high radius in a dense zone will provide heavy neighborhoods.
To solve this issue, the authors proposed to replace the Single Scale Grouping Layer (SSG) by a **Multi Scale Grouping Layer (MSG)**. The idea is to build multiple neighborhoods with different scales (radius) for each representative point, as illustrated in <a href="#figure7">Figure 7.a</a>. A different PointNet is applied to each scale, for the same representative point. The output of the PointNets are then concatenated to form the feature vector of that multi-scale neighborhood. 

However, it is expensive to apply different PointNet architectures for each scale at each SA layer. Another efficient solution has been proposed : **Multi Resolution Grouping layer (MRG)**. The idea is to build neighborhoods with a given fixed radius. Then, a PointNet is applied to subsets of this neighborhood to build feature vectors. Then a second PointNet aggregate those feature vectors into one feature vector, on one hand; and on the other hand, this second PointNet is also applied to the original points of the neighborhood. The two vectors are then concatenated. If the neighborhood is rich, the first vector in the concatenation will provide a fine-grained encoding of this rich neighborhood. On the contrary, if the neighborhood is very sparse, the first encoding will be less reliable and the second encoding will provide a richer information about this neighborhood.

### PointNet++ : Experiments

#### Eperiments on Classification task
In this experiment PointNet++ was trained and evaluated on ModelNet40.


<center>
<div style=" display: table; width: 45%">
<div markdown="1" style="font-size: 0.6em; width: 30%; align-self: center;  vertical-align: middle; display:table-cell;">

|Method|Input| Accuracy %|
|Subvolume <a href="#references">[6]</a>|Voxel grid| 89.2|
|MVCNN <a href="#references">[7]</a> | images | 90.1 |
|PointNet <a href="#references">[1]</a> | point cloud | 89.2 |
|PointNet++ <a href="#references">[2]</a> | point cloud | 90.7 |
|**PointNet++ (xyz + normal vectors) <a href="#references">[2]</a>** | point cloud | **91.9** |


</div>

</div>
<center style="font-style: initial;"><b>Table 3</b> : Classification on ShapeNet.</center>
</center>

PointNet++ outperformed the benchmark at the time of its publication in the classification task.

#### Experiments on Segmentation Task
 PointNet++ was trained, evaluated and compared to other benchmark methods on Scannet dataset <a href="#references">[12]</a> in the semantic segmentation task. This dataset is made of 1513 indoor scans. 
<br>
In this task, the authors did not measure the performance of their model using mIOU score. They rather used an accuracy metric on a per-voxel basis as in **<a href="#references">[12]</a>**. Also, the authors conducted a second experiment with a new dataset based on ScanNet : they called this dataset ScanNet non-uniform. To generate this dataset, the authors used a virtual camera placed in the center of each room. Then orientating this camera in eight (08) different directions, they generate a "visible point cloud" in each direction. In fact, they use an image plane to cast rays from camera to the scene in order to select the visible points. That way, eight new synthetic scans can be generated from a single room, each scan being an observation of the scene from a given point view. The scans are of good quality near the camera and sparser the further we get from the camera. Those synthetic scans can be viewed as realistic non uniform scans.

<center>
<div style=" display: table; width: 75%">
  <figure  style="width:45%; margin:0; align-self: center;  vertical-align: middle; display:table-cell">
  <img src="{{ '/assets/img/scannetUnif.PNG' | prepend: site.baseurl }}" alt="" style=""> 

</figure>
  <figure  style="width:45%; margin:0;  align-self: center;  vertical-align: middle; display:table-cell;">
  <img src="{{ '/assets/img/semanticScannetPointNet++.PNG' | prepend: site.baseurl }}" alt="" style=""> 
  
</figure>
</div>
<p style="font-style: initial;"><b>Figure 8</b>. Left image represents an example of Non-Uniform scan generation. The histogram represents the performance of different models on both datasets. "Ours" refers to PointNet++ and "DP" refers to dropout : some random points are dropped from the point clouds during the training. This can be seen as a data augmentation technique and also a leverage to build a more robust model.</p>
</center>
<br>


<center>
<div style=" display: table; width: 60%">
<div markdown="1" style="font-size: 0.6em; width: 30%; align-self: center;  vertical-align: middle; display:table-cell;">

|**Models**|Yi <a href="#references">[8]</a>| 3DCNN <a href="#references">[13]</a>| PointNet <a href="#references">[1]</a> | **PointNet++ <a href="#references">[2]</a>** |
|**mIOU**|81.4|79.4| 83.7 | **85.1** |

</div>
</div>
<center style="font-style: initial;"><b>Table 4</b> : Part Segmentation on ShapeNet.</center>
</center>
<br>

The authors also studied PointNet++ on the part segmentation task on ShapeNet dataset. Once again, PointNet++ outperformed some of the benchmark methods at that time.

#### Further Analysis : Robustness to sampling density
PointNet++ in its initial configuration (Single Scale Grouping (SSG) in SA layer) is not robust to sampling density. To fix this issue, the authors proposed to replace the SSG by a <b>MSG</b> (or <b>MRG</b> for efficiency). To assess the improvements brought by those proposals, the authors evaluated different architectures on datasets with different densities per objects, on the classification task. PointNet vanilla refers to the version of PointNet without the alignment Network. "Ours" refers to PointNet++. "DP" means "Dropout" : during the training, some points are randomly dropped from the point clouds.
<br><br>
First, dropping points during training is beneficial : it helps the model be robust to the point cloud density. For example, we can clearly see the gap between PointNet Vanilla and PointNet Vanilla (DP) for different densities.<br>
Second, PointNet++ MSG (DP) outperforms PointNet++ SSG (DP) proving that the Multi Scale Grouping is an important module for PointNet++ too trully be robust to varying densities. In fact, PointNet+ SSG performance quickly drops when we evaluate the model on lower densities shapes. 
<center>
<div>
  <figure  style="width:80%; margin:0;">
  <img src="{{ '/assets/img/robustnessPointNet++.PNG' | prepend: site.baseurl }}" alt="" style=""> 
  <center style="font-style: initial;"><b>Figure 9</b>. Left: Point cloud with random point dropout. Right: Curve showing advantage of MSG/MRG in dealing with varying densities in point clouds</center>
</figure>
</div>
</center>
<br>
<br>

PointNet++ MRG is a good compromise between good performance and efficiency. Further experiments comparing the runtime of the different versions of PointNet and PointNet++ shows that:
* **PointNet++ MSG** doubles the runtime of **PointNet++ SSG** (**163.2 ms vs 82.4 ms**).
* **PointNet++ MRG** have approximatly the same runtime (**87.0 ms**) than PointNet++ SSG, while displaying a better performance on non uniform scans. 

### Conclusion

PointNet and PointNet++ are deep learning methods able to process raw point clouds. Thus, they are more memory efficient compared to voxel-based methods. PointNet has an ability to build a feature vector representing the entire point cloud, and extract the most important points that summarize the shape : critical points. PointNet++, thanks to its hierarchical architecture, is more local orientated, and is able to identiffy and understand local structures in complex scenes.


### References
<br>

<textarea id="bibtex_input" style="display:none;">

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
  pos={2}
}

@INPROCEEDINGS{modelnet,
  author={ {Zhirong Wu} and S. {Song} and A. {Khosla} and  {Fisher Yu} and  {Linguang Zhang} and  {Xiaoou Tang} and J. {Xiao}},
  booktitle={2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}, 
  title={3D ShapeNets: A deep representation for volumetric shapes}, 
  year={2015},
  volume={},
  number={},
  pages={1912-1920},
  doi={10.1109/CVPR.2015.7298801},
pos={3}}

  @article{sph,
author = {Kobbelt, Leif and Schrder, P. and Kazhdan, Michael and Funkhouser, Thomas and Rusinkiewicz, Szymon},
year = {2003},
month = {07},
pages = {},
title = {Rotation Invariant Spherical Harmonic Representation of 3D Shape Descriptors},
volume = {vol. 43},
journal = {Proc 2003 Eurographics},
pos={4}
}

@INPROCEEDINGS{voxnet,
  author={D. {Maturana} and S. {Scherer}},
  booktitle={2015 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, 
  title={VoxNet: A 3D Convolutional Neural Network for real-time object recognition}, 
  year={2015},
  volume={},
  number={},
  pages={922-928},
  doi={10.1109/IROS.2015.7353481},
pos={5}}


  @article{subvolume,
  author    = {Charles Ruizhongtai Qi and
               Hao Su and
               Matthias Nie{\ss}ner and
               Angela Dai and
               Mengyuan Yan and
               Leonidas J. Guibas},
  title     = {Volumetric and Multi-View CNNs for Object Classification on 3D Data},
  journal   = {CoRR},
  volume    = {abs/1604.03265},
  year      = {2016},
  url       = {http://arxiv.org/abs/1604.03265},
  archivePrefix = {arXiv},
  eprint    = {1604.03265},
  timestamp = {Mon, 13 Aug 2018 16:46:34 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/QiSNDYG16.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org},
pos={6}
}

@INPROCEEDINGS{mvcnn,  author={H. {Su} and S. {Maji} and E. {Kalogerakis} and E. {Learned-Miller}},  booktitle={2015 IEEE International Conference on Computer Vision (ICCV)},   title={Multi-view Convolutional Neural Networks for 3D Shape Recognition},   year={2015},  volume={},  number={},  pages={945-953},  doi={10.1109/ICCV.2015.114},
pos={7}}


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
pos={8}
}

@INPROCEEDINGS{stanford3D,  author={I. {Armeni} and O. {Sener} and A. R. {Zamir} and H. {Jiang} and I. {Brilakis} and M. {Fischer} and S. {Savarese}},  booktitle={2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},   title={3D Semantic Parsing of Large-Scale Indoor Spaces},   year={2016},  volume={},  number={},  pages={1534-1543},  doi={10.1109/CVPR.2016.170},
pos={9}}

@article{LSTM,
  title={Understanding LSTM--a tutorial into Long Short-Term Memory Recurrent Neural Networks},
  author={Staudemeyer, Ralf C and Morris, Eric Rothstein},
  journal={arXiv preprint arXiv:1909.09586},
  year={2019},
pos={10}
}

@misc{ordermatters,
      title={Order Matters: Sequence to sequence for sets}, 
      author={Oriol Vinyals and Samy Bengio and Manjunath Kudlur},
      year={2016},
      eprint={1511.06391},
      archivePrefix={arXiv},
      primaryClass={stat.ML},
pos={11}
}
@misc{scannet,
      title={ScanNet: Richly-annotated 3D Reconstructions of Indoor Scenes}, 
      author={Angela Dai and Angel X. Chang and Manolis Savva and Maciej Halber and Thomas Funkhouser and Matthias Nießner},
      year={2017},
      eprint={1702.04405},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
pos={12}
}

@misc{3DCnn,
      title={Spectral Networks and Locally Connected Networks on Graphs}, 
      author={Joan Bruna and Wojciech Zaremba and Arthur Szlam and Yann LeCun},
      year={2014},
      eprint={1312.6203},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
pos={13}
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

