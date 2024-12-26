# Cartesian Genetic Programming for Multi-Objective Neural Architecture Search

This repository contains the implementation of the algorithms presented in the following research papers:

## Publications

1. **"Continuous Cartesian Genetic Programming based representation for multi-objective neural architecture search"**  
   *Authors:* Cosijopii Garcia-Garcia, Alicia Morales-Reyes, Hugo Jair Escalante  
   *Published in:* *Applied Soft Computing*, 2023  
   *Abstract:* We propose a novel neural architecture search (NAS) approach for the challenge of designing convolutional neural networks (CNNs) that achieve a good tradeoff between complexity and accuracy. We rely on Cartesian genetic programming (CGP) and integrated real-based and block-chained CNN representation, for optimization using multi-objective evolutionary algorithms (MOEAs) in the continuous domain. We introduce two variants, CGP-NASV1 and CGP-NASV2, which differ in the granularity of their respective search spaces. To evaluate the proposed algorithms, we utilized the non-dominated sorting genetic algorithm II (NSGA-II) on the CIFAR-10, CIFAR-100,and SVHN datasets. Additionally, we extended the empirical analysis while maintaining the same solution representation to assess other searching techniques such as differential evolution (DE), the multi-objective evolutionary algorithm based on decomposition (MOEA/D), and the S metric selection evolutionary multi-objective algorithm (SMS-EMOA). The experimental results demonstrate that our approach exhibits competitive classification performance and model complexity compared to state-of-the-art methods.
   
   *DOI:* [10.1016/j.asoc.2023.110788](https://doi.org/10.1016/j.asoc.2023.110788)  
   

3. **"Progressive Self-supervised Multi-objective NAS for Image Classification"**  
   *Authors:* Cosijopii Garcia-Garcia, Alicia Morales-Reyes, Hugo Jair Escalante  
   *Published in:* *Applications of Evolutionary Computation*, Lecture Notes in Computer Science, vol 14635, 2024  
   *Abstract:* We introduce a novel progressive self-supervised framework for neural architecture search. Our aim is to search for competitive, yet significantly less complex, generic CNN architectures that can be used for multiple tasks (i.e., as a pretrained model). This is achieved through cartesian genetic programming (CGP) for neural architecture search (NAS). Our approach integrates self-supervised learning with a progres- sive architecture search process. This synergy unfolds within the continu- ous domain which is tackled via multi-objective evolutionary algorithms (MOEAs). To empirically validate our proposal, we adopted a rigorous evaluation using the non-dominated sorting genetic algorithm II (NSGA- II) for the CIFAR-100, CIFAR-10, SVHN and CINIC-10 datasets. The experimental results showcase the competitiveness of our approach in relation to state-of-the-art proposals concerning both classification per- formance and model complexity. Additionally, the effectiveness of this method in achieving strong generalization can be inferred.

   *DOI:* [10.1007/978-3-031-56855-8_11](https://doi.org/10.1007/978-3-031-56855-8_11)  
 
5. **"Speeding up the Multi-objective NAS Through Incremental Learning"**  
   *Authors:* Cosijopii Garcia-Garcia, Alicia Morales-Reyes, Hugo Jair Escalante  
   *Published in:* *Applications of Evolutionary Computation*, Lecture Notes in Computer Science, vol 14635, 2024  
   *Abstract:* Deep neural networks (DNNs), particularly convolutional neural networks (CNNs), have garnered significant attention in recent years for addressing a wide range of challenges in image processing and computer vision. Neural architecture search (NAS) has emerged as a crucial field aiming to automate the design and configuration of CNN models. In this paper, we propose a novel strategy to speed up the performance estimation of neural architectures by gradually increasing the size of the training set used for evaluation as the search progresses. We evaluate this approach using the CGP-NASV2 model, a multi-objective NAS method, on the CIFAR-100 dataset. Experimental results demonstrate a notable acceleration in the search process, achieving a speedup of 4.6 times compared to the baseline. Despite using limited data in the early stages, our proposed method effectively guides the search towards competitive architectures. This study highlights the efficacy of leveraging lower-fidelity estimates in NAS and paves the way for further research into accelerating the design of efficient CNN architectures.

   *DOI:* [10.1007/978-3-031-75543-9_1](https://doi.org/10.1007/978-3-031-75543-9_1)  


## Repository Structure
 

- **`CGPNASV2.py`**: Script implementing the CGP-NASV2 algorithm.
- **`CGPNASW_SS.py`**: Script for the progressive self-supervised framework.
- **`CGPNASpeed.py`**: Script for the Paper Speeding up the Multi-objective NAS Through Incremental Learning.
- **`RunSolution.py`**: Script to execute the solutions obtained from the NAS process.
- **`Evolution/`**: Directory containing modules related to evolutionary operations such as crossover and mutation.
- **`Utils/`**: Utility functions for Datasets
- **`CGPNASWSS/`**:  Directory containing modules related to he progressive self-supervised framework.
- **`requirements.txt`**: List of Python dependencies required to run the code.


Our codebase CGP is inspired by Suganuma work in the repository https://github.com/sg-nm/cgp-cnn-PyTorch.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Cosijopiii/CGPNAS.git
   cd CGPNAS
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt



## Citation
If you find this repository useful in your research, please consider citing the following papers:
```bibtex
@article{Garcia-Garcia2023,
  title = {Continuous Cartesian Genetic Programming based representation for multi-objective neural architecture search},
  journal = {Applied Soft Computing},
  volume = {147},
  pages = {110788},
  year = {2023},
  issn = {1568-4946},
  month = sep,
  doi = {10.1016/j.asoc.2023.110788},
  author = {Garcia-Garcia, Cosijopii and Morales-Reyes, Alicia and Escalante, Hugo Jair},
  keywords = {Neural architecture search, Cartesian genetic programming, Convolutional neural network, Multi-objective optimization}
}


@article{Garcia-Garcia2024A,
  author = {Garcia-Garcia, Cosijopii and Morales-Reyes, Alicia and Escalante, Hugo Jair},
  doi = {10.1007/978-3-031-56855-8_11},
  pages = {180--195},
  title = {{Progressive Self-supervised Multi-objective NAS for Image Classification}},
  year = {2024},
  journal = {International Conference on the Applications of Evolutionary Computation (Part of EvoStar)}
}


@article{Garcia-Garcia2024B,
  author = {Garcia-Garcia, Cosijopii and Derbel, Bilel and Morales-Reyes, Alicia and Escalante, Hugo Jair},
  editor = {Mart{\'i}nez-Villase{\~{n}}or, Lourdes and Ochoa-Ruiz, Gilberto},
  title = {Speeding up the Multi-objective NAS Through Incremental Learning},
  booktitle = {Advances in Soft Computing},
  journal = {23rd Mexican International Conference on Artificial Intelligence},
  year = {2025},
  publisher = {Springer Nature Switzerland},
  address = {Cham},
  pages = {3--15},
  isbn = {978-3-031-75543-9}
}
