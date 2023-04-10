<a name="readme-top"></a>

<!-- PROJECT SHIELDS -->
<div align="center">
  
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]
[![Twitter][twitter-shield]][twitter-url] 

</div>

<h3 align="center">ML Classifier Comparison</h3>

  <p align="center">
    Comparing six ML classifiers on COVID-19 data from Brazil with the help of a 2021 paper. <br />
    Additional methods to improve efficiency and effectiveness are included.
    <br />
    <br />
    <a href="https://github.com/keatonrproud/ML_classifier_comparison/issues">Report Bug</a>
    ·
    <a href="https://github.com/keatonrproud/ML_classifier_comparison/issues">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

Six ML classifiers were compared to see which best predicted COVID-19 outcomes from Brazil cases with the help of a 2021 paper by Fernanda Sumika Hojo De Souza, Natalia Satchiko Hojo-Souza, Edimilson Batista Dos Santos, Cristiano Maciel Da Silva, and Daniel Ludovico Guidoni. Further attempts to improve effectiveness and efficiency of the models are included.

<br />

Data was accessed from https://coronavirus.es.gov.br/painel-covid-19-es. Only data until March 29, 2021 was used and stored in this repo, but data updated to today is available at the link provided.

### Built With

[![Python][python-shield]][python-url]
[![Jupyter Notebook][jupyter-shield]][jupyter-url]
[![PyCharm][pycharm-shield]][pycharm-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

If you want an overview of the code and the outputs, [project.ipynb](https://github.com/keatonrproud/ML_classifier_comparison/blob/main/project.ipynb) makes for easy viewing. 
Alternatively, clone the repo and access the scripts locally.

### Prerequisites

You'll need several packages to run the scripts directly -- _[imblearn](https://imbalanced-learn.org/stable/), [matplotlib](https://matplotlib.org/), [numpy](https://numpy.org/), [pandas](https://pandas.pydata.org/), [seaborn](https://seaborn.pydata.org/), [sklearn](https://scikit-learn.org/stable/), [sklearnex](https://intel.github.io/scikit-learn-intelex/)_.

Ensure you have all packages installed. _Note that speed improvements from sklearnex are only possible when using Intel processors._


### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/keatonrproud/ML_classifier_comparison.git
   ```
2. Install any missing packages
   ```sh
   pip install missing_package
   ```


<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- CONTRIBUTING -->
## Contributing

If you have a suggestion, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement". All feedback is appreciated!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* [Fernanda Sumika Hojo De Souza, Natalia Satchiko Hojo-Souza, Edimilson Batista Dos Santos, Cristiano Maciel Da Silva, Daniel Ludovico Guidoni](https://www.frontiersin.org/articles/10.3389/frai.2021.579931/full) for the excellent paper.
* [University of Bologna](https://www.unibo.it/en) for the project idea.
* [Scikit-Learn Docs](https://scikit-learn.org/)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Keaton Proud - [Website](https://keatonrproud.github.io) - [LinkedIn](https://linkedin.com/in/keatonrproud) - [Twitter](https://twitter.com/keatonrproud) - keatonrproud@gmail.com

Project Link: [https://github.com/keatonrproud/ML_classifier_comparison](https://github.com/keatonrproud/ML_classifier_comparison)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- LINKS & IMAGES -->
[contributors-shield]: https://img.shields.io/github/contributors/keatonrproud/ML_classifier_comparison.svg?style=for-the-badge
[contributors-url]: https://github.com/keatonrproud/ML_classifier_comparison/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/keatonrproud/ML_classifier_comparison.svg?style=for-the-badge
[forks-url]: https://github.com/keatonrproud/ML_classifier_comparison/network/members
[stars-shield]: https://img.shields.io/github/stars/keatonrproud/ML_classifier_comparison.svg?style=for-the-badge
[stars-url]: https://github.com/keatonrproud/ML_classifier_comparison/stargazers
[issues-shield]: https://img.shields.io/github/issues/keatonrproud/ML_classifier_comparison.svg?style=for-the-badge
[issues-url]: https://github.com/keatonrproud/ML_classifier_comparison/issues
[license-shield]: https://img.shields.io/github/license/keatonrproud/ML_classifier_comparison.svg?style=for-the-badge
[license-url]: https://github.com/keatonrproud/ML_classifier_comparison/blob/main/license
[linkedin-shield]: https://img.shields.io/badge/linkedin-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white
[linkedin-url]: https://linkedin.com/in/keatonrproud
[twitter-shield]: https://img.shields.io/badge/Twitter-%231DA1F2.svg?style=for-the-badge&logo=Twitter&logoColor=white
[twitter-url]: https://twitter.com/keatonrproud
[python-shield]: https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54
[python-url]: https://python.org/
[jupyter-shield]: https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white
[jupyter-url]: https://jupyter.org/
[pycharm-shield]: https://img.shields.io/badge/pycharm-143?style=for-the-badge&logo=pycharm&logoColor=black&color=black&labelColor=green
[pycharm-url]: [https://jupyter.org/](https://www.jetbrains.com/pycharm/)
