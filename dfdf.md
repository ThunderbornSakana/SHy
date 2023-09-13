# Fall 2023 COMP 576 Assignment 0 Report
Student: Leisheng Yu

## Task 1
```bash
     active environment : base
    active env location : /Users/leishengyu/opt/anaconda3
            shell level : 1
       user config file : /Users/leishengyu/.condarc
 populated config files : /Users/leishengyu/.condarc
          conda version : 4.14.0
    conda-build version : 3.21.5
         python version : 3.9.7.final.0
       virtual packages : __osx=10.16=0
                          __unix=0=0
                          __archspec=1=x86_64
       base environment : /Users/leishengyu/opt/anaconda3  (writable)
      conda av data dir : /Users/leishengyu/opt/anaconda3/etc/conda
  conda av metadata url : None
           channel URLs : https://repo.anaconda.com/pkgs/main/osx-64
                          https://repo.anaconda.com/pkgs/main/noarch
                          https://repo.anaconda.com/pkgs/r/osx-64
                          https://repo.anaconda.com/pkgs/r/noarch
          package cache : /Users/leishengyu/opt/anaconda3/pkgs
                          /Users/leishengyu/.conda/pkgs
       envs directories : /Users/leishengyu/opt/anaconda3/envs
                          /Users/leishengyu/.conda/envs
               platform : osx-64
             user-agent : conda/4.14.0 requests/2.26.0 CPython/3.9.7 Darwin/22.5.0 OSX/10.16
                UID:GID : 501:20
             netrc file : /Users/leishengyu/.netrc
           offline mode : False
```


## Task 2
```bash
import numpy as np
a = np.array([[1., 2., 3.], [4., 5., 6.]])
a.ndim
```
2
```bash
a.size
```
6
```bash
a.size
```
(2, 3)
```bash
b = np.array([[1., 2., 3.], [4., 5., 6.]])
c = np.array([[1., 2., 3.], [4., 5., 6.]])
d = np.array([[1., 2., 3.], [4., 5., 6.]])
np.block([[a, b], [c, d]])
```
array([[1., 2., 3., 1., 2., 3.],
       [4., 5., 6., 4., 5., 6.],
       [1., 2., 3., 1., 2., 3.],
       [4., 5., 6., 4., 5., 6.]])


## Task 3
![task3_fig](https://github.com/ThunderbornSakana/SHy/assets/84387542/a529b5cd-69c5-436b-a725-59666478be3d)


## Task 4
```bash
x = np.arange(0, 10*np.pi, 0.1)
y = np.sin(x)
plt.plot(x, y)
plt.show()
```
![task4_fig](https://github.com/ThunderbornSakana/SHy/assets/84387542/e0e3816a-3af2-456f-8181-41570646179f)


## Task 5
[https://github.com/ThunderbornSakana](https://github.com/ThunderbornSakana)


## Task 6
[https://github.com/ThunderbornSakana/COMP576](https://github.com/ThunderbornSakana/COMP576)
