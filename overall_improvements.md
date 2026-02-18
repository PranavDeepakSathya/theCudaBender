matmul V3 (persistance, breaks when BK_stages = 3)
we are using a global stage index based on the linear iteration of the block tiles's lifetimes 
and token based. 
Which of the above (the former or the latter) breaks it? 
(one has to really understand synchronization and pipelining to get this)

Next, we need to clean up our kernels A LOT, and switch to the ptx wrappers. 
and use the 'precompute address' tricks. In fact, hand planning our kernels might be a great idea
and somehow escape the fucking soup of shit we're doing right now. 




