import torch
import torch.nn as nn

E=eval(input('Enter value of elasticity : '))#210e9
I=eval(input('enter moment of inertia : '))#1/12
L_phsyical=eval(input('Enter length of beam :'))#8.0
q_physical=eval(input('enter value of distributed load per m :'))#10.0 #distributed load
EI_physical=E*I
beam_type=input('enter type of beam simply or cantilever:')


# bringing back the normalized term to normal scale
x_scale=L_phsyical
d_scale=abs(q_physical)*L_phsyical**4/EI_physical

#we have to make the length,flexural rigidity and loading normalized because PINNs model operate 
#in normalized domain
L_norm=1.0
EI_norm=1.0
q_norm=1.0



#Hyperparameters
N_f=500 #no. of points we assume on the beam
lr=0.001
loss_weight_bc=10
loss_weight_phy=1
no_adam_steps=5000

#PINN Model
class PINN(nn.Module):
    def __init__(self,):
        super(PINN,self).__init__()

        self.linear1=nn.Linear(1,64)
        self.tanh1=nn.Tanh()
        self.linear2=nn.Linear(64,64)
        self.tanh2=nn.Tanh()
        self.linear3=nn.Linear(64,1)
# we employ Xavier intialization for better convergence for a smooth intial functions
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias,0)
    
    def forward(self,x):
        out=self.linear1(x)
        out=self.tanh1(out)
        out=self.linear2(out)
        out=self.tanh2(out)
        out=self.linear3(out)
        return out

model=PINN()
optimizer_Adam=torch.optim.Adam(model.parameters(),lr=lr)

optimizer_lbfgs = torch.optim.LBFGS(
    model.parameters(), 
    lr=1.0,  # L-BFGS learning rate
    max_iter=20, # Max iterations *per step*
    tolerance_grad=1e-10,
    tolerance_change=1e-13,
    history_size=50,
    line_search_fn="strong_wolfe" )

#Loss function

def loss():
    # Boundary condition points
    x_bc_0=torch.tensor([[0.0]],requires_grad=True)
    x_bc_1=torch.tensor([[L_norm]],requires_grad=True)

    # Points on the beam
    torch.manual_seed(42)
    x_c=torch.rand(N_f,1,requires_grad=True)

    if (beam_type=='simply'):
        # deflection on Boundary x=0 is y=0
        d_0=model(x_bc_0) # d(0)
        dx_0=torch.autograd.grad(d_0,x_bc_0,create_graph=True)[0] #d(d_0)/dx
        dxx_0=torch.autograd.grad(dx_0,x_bc_0,create_graph=True)[0] #d^2(d_0)/dx^2
    
        # deflection on Boundary x=L is y=0
        d_L=model(x_bc_1)
        dx_L=torch.autograd.grad(d_L,x_bc_1,create_graph=True)[0] #d(d_L)/dx
        dxx_L=torch.autograd.grad(dx_L,x_bc_1,create_graph=True)[0] #d^2(d_L)/dx^2
    
        # loss calculation at boundary condition 
        # all the boundary losses should be zero
        loss_bc_1=d_0**2 
        loss_bc_2=dxx_0**2
        loss_bc_3=d_L**2
        loss_bc_4=dxx_L**2
        # Total loss at the boundary
        L_total_bc=loss_bc_1+loss_bc_2+loss_bc_3+loss_bc_4
    
        # deflection at any point in between (0,L)
        d_x=model(x_c)
        d1_x_c=torch.autograd.grad(d_x,x_c,torch.ones_like(d_x),create_graph=True,retain_graph=True)[0]
        d2_x_c=torch.autograd.grad(d1_x_c,x_c,torch.ones_like(d1_x_c),create_graph=True,retain_graph=True)[0]
        d3_x_c=torch.autograd.grad(d2_x_c,x_c,torch.ones_like(d2_x_c),create_graph=True,retain_graph=True)[0]
        d4_x_c=torch.autograd.grad(d3_x_c,x_c,torch.ones_like(d3_x_c),create_graph=True,retain_graph=True)[0] 

    else:
        # Boundary condition at x=0 y=0 and dy/dx=0
        d_0=model(x_bc_0) # d(0)
        dx_0=torch.autograd.grad(d_0,x_bc_0,create_graph=True)[0] #d(d_0)/dx
        # Boundary condition at x=L d2y/dx2=0 and d3y/dx3=0
        d_L=model(x_bc_1)
        dx_L=torch.autograd.grad(d_L,x_bc_1,create_graph=True)[0] #d(d_L)/dx
        dxx_L=torch.autograd.grad(dx_L,x_bc_1,create_graph=True)[0] #d^2(d_L)/dx^2
        dxxx_L=torch.autograd.grad(dxx_L,x_bc_1,create_graph=True)[0] #d^2(d_L)/dx^2
        # loss calculation at boundary condition 
        # all the boundary losses should be zero
        loss_bc_1=d_0**2 
        loss_bc_2=dx_0**2
        loss_bc_3=dxxx_L**2
        loss_bc_4=dxx_L**2
        # Total loss at the boundary
        L_total_bc=loss_bc_1+loss_bc_2+loss_bc_3+loss_bc_4
        # deflection at any point in between (0,L)
        d_x=model(x_c)
        d1_x_c=torch.autograd.grad(d_x,x_c,torch.ones_like(d_x),create_graph=True,retain_graph=True)[0]
        d2_x_c=torch.autograd.grad(d1_x_c,x_c,torch.ones_like(d1_x_c),create_graph=True,retain_graph=True)[0]
        d3_x_c=torch.autograd.grad(d2_x_c,x_c,torch.ones_like(d2_x_c),create_graph=True,retain_graph=True)[0]
        d4_x_c=torch.autograd.grad(d3_x_c,x_c,torch.ones_like(d3_x_c),create_graph=True,retain_graph=True)[0] 

    Residual=d4_x_c-1
    L_physics=torch.mean(Residual**2)

    # Total Loss=L_bc+L_phy
    L_total=L_physics*loss_weight_phy+L_total_bc*loss_weight_bc

    return L_total,L_physics,L_total_bc

# we are gonna use the hybrid of Adam+L-BFGS optimizer for the training
print('----------------------starting phase 1 of training--------------------------------------')
model.train()
for i in range(no_adam_steps):
    optimizer_Adam.zero_grad()
    L_total,L_unweighted_phy,L_unweighted_bc=loss()
# if there is NAN value then the loop is not run
    if torch.isnan(L_total):
        print(f"NaN detected at step {i}! Stopping Adam.")
        break

    L_total.backward() #backpropogation

# gradient clipping is done in PINN model so that while calculating higher order derivative the 
# gradient value dont shoot up and weights become too large that the training becomes unstable 
# also it only reduce the magnitude but preserve the direction
    torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)

    optimizer_Adam.step() #update model weight

    if i%500==0:
        print(f"Step {i:5d} | Loss: {L_total.item():.6e} | BC: {L_unweighted_bc.item():.6e} | Physics: {L_unweighted_phy.item():.6e}")

print("Adam finished.\n")

# --- Phase 2: L-BFGS Optimizer ---
print("--- Starting Phase 2: L-BFGS Optimizer ---")
lbfgs_step = 0

def lbfgs_closure():
    global lbfgs_step

    optimizer_lbfgs.zero_grad()
    L_total,L_unweighted_bc,L_unweighted_phy=loss()

    if torch.isnan(L_total):
        print(f"NaN detected at step {i}! Stopping LBFGS.")
        return L_total
    
    L_total.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)

    if lbfgs_step % 10 == 0:
        print(f"Step {lbfgs_step:5d} | Loss: {L_total.item():.6e} | BC: {L_unweighted_bc.item():.6e} | Physics: {L_unweighted_phy.item():.6e}")
    
    lbfgs_step += 1
    
    return L_total

# Run L-BFGS
model.train()
try:
    # optimizer.step() is the only call needed for L-BFGS.
    # It will call the 'lbfgs_closure' function repeatedly
    # until convergence (or max_iter is reached).
    optimizer_lbfgs.step(lbfgs_closure)
    print("\nL-BFGS completed successfully!")
except RuntimeError as e:
    print(f"\nL-BFGS encountered an error (can be safely ignored if training converged): {e}")

print("\nTraining finished.")

# Saving the trained model 
torch.save(model.state_dict(), "model.pth")

