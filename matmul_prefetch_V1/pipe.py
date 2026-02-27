class OuterJob:
    def __init__(self, n_jobs, n_buffers):
        self.n_jobs = n_jobs
        self.n_buffers = n_buffers
        self.phase = [0] * n_buffers   # one phase per buffer

    def load_into(self, job_idx, buffer_idx):
        old = self.phase[buffer_idx]
        new = old ^ 1
        self.phase[buffer_idx] = new
        return f"O{job_idx}:B{buffer_idx} p{old}→{new}"

    
    def wait_parity(self, buffer_idx, parity):
        current = self.phase[buffer_idx]
        return f"W B{buffer_idx} wait_p{parity} (cur_p{current})"
      
      
class InnerJob:
    def __init__(self, n_jobs, n_buffers):
        self.n_jobs = n_jobs
        self.n_buffers = n_buffers

    def load_into(self, job_idx, buffer_idx):
        return f"I{job_idx}:b{buffer_idx}"

    def compute(self, job_idx, buffer_idx):
        return f"b{buffer_idx}→C{job_idx}"
      
   
def print_step(step, outer=None, inner_load=None, inner_compute=None):
    left  = f"{outer:<18}" if outer else " " * 18
    mid   = f"{inner_load:<10}" if inner_load else " " * 10
    right = f"{inner_compute:<10}" if inner_compute else " " * 10
    print(f"{step:04d} | {left} | {mid} | {right}")


N_outer_jobs = 1024//64
N_outer_stages = 2
N_inner_jobs = 64//16
N_inner_stages = 2

O_JOB = OuterJob(N_outer_jobs, N_outer_stages)
I_JOB = InnerJob(N_inner_jobs, N_inner_stages)

num_logical_iters = (N_outer_stages-1) + (N_inner_stages-1) + (N_outer_jobs*N_inner_jobs)

i = 0

for _ in range(N_outer_stages-1):
  outer_str, inner_load_str, inner_compute_str = None,None,None
  outer_str = O_JOB.load_into(i,i%N_outer_stages)
  print_step(i,outer_str, inner_load_str, inner_compute_str)
  i +=1
  
for q in range(N_inner_stages-1): 
  outer_str, inner_load_str, inner_compute_str = None,None,None
  if (q == 0): 
    outer_str = O_JOB.load_into(i,i%N_outer_stages)
    print(O_JOB.wait_parity(i%N_outer_stages,0))
  #the first wait for the first set of inner loads will be done here. 
  inner_load_str = I_JOB.load_into(q, q%N_inner_stages)
  print_step(i,outer_str, inner_load_str, inner_compute_str)
  i+=1
    
NUM_FULL_ITERS = (N_outer_jobs - N_outer_stages)*N_inner_jobs
print("--------------------------------------------------------------------------------------------------")
for j in range(NUM_FULL_ITERS): 
  if (j > 0) and (j%N_inner_jobs == 0) : print ("-------------------------------------------------------------")
  outer_str, inner_load_str, inner_compute_str = None,None,None
  
  inner_compute_idx = j % N_inner_jobs
  inner_compute_stage = j % N_inner_stages
  inner_load_idx = (j + (N_inner_stages-1)) % N_inner_jobs
  inner_load_stage = (j + (N_inner_stages-1)) % N_inner_stages 
  outer_load_idx = ((j//N_inner_jobs) + (N_outer_stages)) % N_outer_jobs #this modulus is not needed but hey
  outer_load_stage = ((j//N_inner_jobs) + (N_outer_stages)) % N_outer_stages
  
  outer_consume_idx = ((j + (N_inner_stages-1))//N_inner_jobs) 
  outer_consume_stage = outer_consume_idx % N_outer_stages
  outer_consume_cycle = outer_consume_idx // N_outer_stages 
  
  wait_parity = outer_consume_cycle % 2
  
  if (inner_load_idx == 0):
    print(O_JOB.wait_parity(outer_consume_stage,wait_parity))
    outer_str = O_JOB.load_into(outer_load_idx,outer_load_stage)

    #previous outer stage is done. 

  
  inner_load_str = I_JOB.load_into(inner_load_idx, inner_load_stage)
  inner_compute_str = I_JOB.compute(inner_compute_idx, inner_compute_stage)
                      
  print_step(i,outer_str, inner_load_str, inner_compute_str)
  i+=1
  
NUM_EPILOGUE_ITERS = (N_outer_stages-1)*N_inner_jobs 
  
for j in range(NUM_FULL_ITERS, NUM_FULL_ITERS + NUM_EPILOGUE_ITERS): 
  if (j > 0) and (j%N_inner_jobs == 0) : print ("-------------------------------------------------------------")
  outer_str, inner_load_str, inner_compute_str = None,None,None
  
  inner_compute_idx = j % N_inner_jobs
  inner_compute_stage = j % N_inner_stages
  inner_load_idx = (j + (N_inner_stages-1)) % N_inner_jobs
  inner_load_stage = (j + (N_inner_stages-1)) % N_inner_stages 

  outer_consume_idx = ((j + (N_inner_stages-1))//N_inner_jobs) 
  outer_consume_stage = outer_consume_idx % N_outer_stages
  outer_consume_cycle = outer_consume_idx // N_outer_stages 
  
  wait_parity = outer_consume_cycle % 2
  
  if (inner_load_idx == 0):
   print(O_JOB.wait_parity(outer_consume_stage,wait_parity))
    #previous outer stage is done. 

  
  inner_load_str = I_JOB.load_into(inner_load_idx, inner_load_stage)
  inner_compute_str = I_JOB.compute(inner_compute_idx, inner_compute_stage)
                      
  print_step(i,outer_str, inner_load_str, inner_compute_str)
  i+=1

  
