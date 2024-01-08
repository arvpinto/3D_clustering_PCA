<br>
<p align="justify"><b>This tutorial aims to cluster frames in cMD trajectories according to PCA vectors obtained with GROMACS analysis tools. </b></p>

<p align="justify"> It requires a cMD *xtc trajectory file, a *tpr and a *gro file. In the first part, the trajectory is corrected with gmx trjconv and the PCA vectors are calculated. Then, the pca_dbscan_gmm.py uses DBSCAN to identify outliers and performs a clustering analysis with Gaussian mixture models. The script is also able to identify the PCA vectors that correspond to the highest density of frames in a cluster, through a kernel density-based method, and output the 5 frames that are closest to the identified point. In the second part, the frames are extracted from the trajectory to separate *.gro files. </p>

<br>
---
<br>


<br>
<h2> <p align="center"> <b>I - Geometry optimization of the reactant</b> </p></h2>

<br/>

```js
&GLOBAL
    RUN_TYPE GEO_OPT
...
```

---






