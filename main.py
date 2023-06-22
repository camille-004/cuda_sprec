"""Main file."""
from cusprec.data.dataset import BasicDataset

if __name__ == "__main__":
    data = BasicDataset(1024, 512, 10)
    data.plot()
