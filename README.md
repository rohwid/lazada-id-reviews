# **Lazada ID Reviews**

![workflow status](https://github.com/rohwid/lazada-id-reviews/actions/workflows/ci.yaml/badge.svg)
![workflow status](https://github.com/rohwid/lazada-id-reviews/actions/workflows/cd-staging.yaml/badge.svg)
![workflow status](https://github.com/rohwid/lazada-id-reviews/actions/workflows/cd-push-registry.yaml/badge.svg)
![workflow status](https://github.com/rohwid/lazada-id-reviews/actions/workflows/cd-production.yaml/badge.svg)

Steps:
+ Select **Use this template** > **Create a new repository**. This menu is in the top right corner of this repository.
+ Edit `setup.py` and define your the project repository.
    + Edit the [README.md](README.md) file's **workflow status** badge with the name of your repository.
    + Rename `src/MLProject` to your project name.
+ Create virtual environment

    ```bash
    virtualenv .venv -p /usr/bin/python3.10
    ```
  **Note:** You can use any Python version, as long the Python packages in `requirements.txt` are supported. Because of the Python packages in `requirements.txt` were declared without describe the version.
+ Activate the virtual environment

    ```bash
    source .venv/bin/activate
    ```

+ Install package.

    ```bash
    pip install -r requirements.txt
    ```

+ enable the environment variables.

    ```bash
    cp .env.example .env
    ```