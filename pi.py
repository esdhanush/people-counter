import pip

def install_deep_sort_pytorch():
    install_cmd = "pip install git+https://github.com/ZQPei/deep_sort_pytorch.git"
    pip.main([install_cmd])

if __name__ == "__main__":
    install_deep_sort_pytorch()
