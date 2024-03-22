
---

# Contributing to VineCopulas

Welcome to `VineCopulas`! We appreciate your interest in contributing to this project. By contributing, you can help make `VineCopulas` better for everyone. Please take a moment to review the following guidelines before making your contribution.

## Code of Conduct

This project and everyone participating in it is governed by the
[Code of Conduct](https://github.com/VU-IVM/VineCopulas/blob/develop/CODE_OF_CONDUCT.md).
By participating, you are expected to uphold this code. Please report unacceptable behavior
to j.n.claassen@vu.nl.

## Asking questions and reporting Issues

If you encounter any bugs or issues while using `VineCopulas`, please report them by opening an *issue* in the GitHub repository. Be sure to provide detailed information about the problem, such as steps to reproduce it, including operating system and Python version.

If you have any suggestions for improvements, or questions, please also raise an *issue*. 

## Contributing to Code

### Setting Up Your Development Environment

1. Clone the repository to your local machine:

    ```bash
    git clone https://github.com/VU-IVM/VineCopulas.git
    ```

3. Create a new environment to install the library with dependencies, for example using conda:

    ```bash
   conda create -n VineCopulasDev python=3
   conda activate VineCopulasDev
    ```
   Install `VineCopulas` using the following command in the projects main directory

    ```bash
   pip install -e .
    ```

4. Checkout a new branch for your changes from the main branch:

    ```bash
    git checkout -b feature/your-feature-name
    ```

5. Make your changes.

## Submitting a Pull Request

Once you have made your changes and are ready to contribute, follow these steps to submit a pull request:

1. Push your changes back to origin:

    ```bash
    git push origin feature/your-feature-name
    ```

2. Create a pull request to merge your branch into the main branch.

Provide a clear description of your changes in the pull request template.

#### Pull request guidelines
- Write clear and concise commit messages.
- Test your changes thoroughly before submitting a pull request.
- If the pull request adds functionality, the docs should also be updated. Improving documentation helps users better understand how to use `VineCopulas`


## Review Process

All pull requests will undergo a review process before being accepted. Reviewers may provide feedback or request changes to ensure the quality of the codebase.


