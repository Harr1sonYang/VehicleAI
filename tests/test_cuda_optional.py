import pytest


@pytest.mark.cuda
def test_cuda_tensor_operation() -> None:
    torch = pytest.importorskip("torch")
    assert torch.cuda.is_available(), "CUDA test requires a CUDA-enabled runner"
    left = torch.tensor([[1.0, 2.0]], device="cuda")
    right = torch.tensor([[3.0], [4.0]], device="cuda")
    assert (left @ right).item() == pytest.approx(11.0)
