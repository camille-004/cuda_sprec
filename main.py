"""Main file."""
from cusprec.algorithms.omp import OMP
from cusprec.data.dataset import BasicDataset

if __name__ == "__main__":
    data = BasicDataset(1024, 512, 10)
    model = OMP(data)
    model.execute()
    result = model.get_results()
