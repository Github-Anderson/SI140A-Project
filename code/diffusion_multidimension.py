import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load and preprocess data
data = np.load('raw_data_2.npy')
dataset = torch.Tensor(data).float().to(device)
dimension = data.shape[1]
print("The dimension of the data is: ", dimension, " and the number of samples is: ", data.shape[0], 'The shape of the data is: ', data.shape)
# Hyperparameters
num_steps = 100  # Diffusion process time steps
num_epochs = 4000  # Number of training epochs
batch_size = 256  # Batch size

# Create beta schedule (linearly increasing betas)
betas = torch.linspace(1e-5, 0.5e-2, num_steps).to(device)

# Compute alpha and related terms
alphas = 1 - betas
alphas_prod = torch.cumprod(alphas, dim=0)
alphas_prod_p = torch.cat([torch.tensor([1.0], device=device), alphas_prod[:-1]], dim=0)
alphas_bar_sqrt = torch.sqrt(alphas_prod).to(device)
one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod).to(device)

# Noise addition function in diffusion process
def q_x(x_0, t):
    noise = torch.randn_like(x_0).to(device)  # Standard normal noise
    alphas_t = alphas_bar_sqrt[t].unsqueeze(1)  # Shape: (batch_size, 1)
    alphas_l_m_t = one_minus_alphas_bar_sqrt[t].unsqueeze(1)  # Shape: (batch_size, 1)
    return alphas_t * x_0 + alphas_l_m_t * noise

# Checkpoint functions
def save_checkpoint(model, optimizer, epoch, loss, filename):
    """Save model and optimizer state."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename}")

def load_checkpoint(model, optimizer, filename, device):
    """Load model and optimizer state."""
    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded from {filename} at epoch {epoch} with loss {loss}")
    return epoch, loss

# Define the MLP Diffusion model
class MLPDiffusion(nn.Module):
    def __init__(self, n_steps, dimension, num_units=128):
        super(MLPDiffusion, self).__init__()
        self.step_embedding = nn.Embedding(n_steps, num_units)
        self.net = nn.Sequential(
            nn.Linear(dimension + num_units, num_units),
            nn.ReLU(),
            nn.Linear(num_units, num_units),
            nn.ReLU(),
            nn.Linear(num_units, num_units),
            nn.ReLU(),
            nn.Linear(num_units, dimension),  # Output dimensions (x1, x2, x3)
        )

    def forward(self, x, t):
        t_embedding = self.step_embedding(t)  # Shape: (batch_size, num_units)
        x = torch.cat([x, t_embedding], dim=1)  # Concatenate along feature dimension
        return self.net(x)

# Loss function: Mean Squared Error between predicted and true noise
def diffusion_loss_fn(model, x_0, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, n_steps):
    batch_size = x_0.shape[0]

    # Randomly sample a timestep t for each example in the batch
    t = torch.randint(0, n_steps, size=(batch_size,), device=device)

    # Compute noisy input x(t)
    a = alphas_bar_sqrt[t].unsqueeze(1)  # Shape: (batch_size, 1)
    e = torch.randn_like(x_0).to(device)  # Shape: (batch_size, 3)
    aml = one_minus_alphas_bar_sqrt[t].unsqueeze(1)  # Shape: (batch_size, 1)
    x = a * x_0 + aml * e  # Shape: (batch_size, 3)

    # Predict the noise using the model
    output = model(x, t)  # Shape: (batch_size, 3)

    # Compute the loss (MSE between true noise and predicted noise)
    return ((e - output) ** 2).mean()

# Training function
def train_model(model, dataset, optimizer, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, num_epochs):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for batch_x in dataloader:
            batch_x = batch_x.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Compute loss
            loss = diffusion_loss_fn(model, batch_x, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, num_steps)

            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_x.size(0)

        epoch_loss /= len(dataloader.dataset)

        if (epoch + 1) % 100 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.6f}")

    return epoch_loss

# Sampling functions
def p_sample(model, x, t, betas, one_minus_alphas_bar_sqrt, device):
    """
    Perform one step of the reverse diffusion process.
    """
    # Compute coefficients and reshape to (batch_size, 1) for broadcasting
    coeff = (betas[t] / one_minus_alphas_bar_sqrt[t]).unsqueeze(1)  # Shape: (1000, 1)
    inv_sqrt_one_minus_betas = (1 / torch.sqrt(1 - betas[t])).unsqueeze(1)  # Shape: (1000, 1)
    sqrt_betas = torch.sqrt(betas[t]).unsqueeze(1)  # Shape: (1000, 1)
    
    # Predict the noise using the model
    eps_theta = model(x, t)  # Shape: (1000, 3)
    
    # Compute the mean
    mean = inv_sqrt_one_minus_betas * (x - coeff * eps_theta)  # Shape: (1000, 3)
    
    # Sample noise for the current step
    z = torch.randn_like(x).to(device)  # Shape: (1000, 3)
    
    # Compute the sample
    sample = mean + sqrt_betas * z  # Shape: (1000, 3)
    
    return sample

def p_sample_loop(model, shape, n_steps, betas, one_minus_alphas_bar_sqrt, device):
    """
    Generate samples by iteratively applying p_sample.
    """
    model.eval()
    with torch.no_grad():
        cur_x = torch.randn(shape).to(device)  # Initialize with random noise
        x_seq = [cur_x]
        for i in reversed(range(n_steps)):
            t = torch.full((shape[0],), i, dtype=torch.long, device=device)  # Shape: (1000,)
            cur_x = p_sample(model, cur_x, t, betas, one_minus_alphas_bar_sqrt, device)  # Shape: (1000, 3)
            x_seq.append(cur_x)
    return x_seq

# Initialize model, optimizer, and train
model = MLPDiffusion(num_steps, dimension, num_units=128).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
checkpoint_filename = 'checkpoint0.ckpt'

# Train the model
final_loss = train_model(model, dataset, optimizer, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, num_epochs)

# Save the checkpoint
save_checkpoint(model, optimizer, num_epochs, final_loss, checkpoint_filename)

# Load the checkpoint (if needed)
# load_checkpoint(model, optimizer, checkpoint_filename, device)

# Generate samples
generated_samples_seq = p_sample_loop(model, (1000, dimension), num_steps, betas, one_minus_alphas_bar_sqrt, device)

# Extract the final generated samples
generated_samples = generated_samples_seq[-1].cpu().detach().numpy()
np.save('toy.npy', generated_samples)
