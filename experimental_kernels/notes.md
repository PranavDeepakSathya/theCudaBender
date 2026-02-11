This kernel is fragile for apw_m = 2 for some reason, it is in fact sensitive 
for all small sizes of apw_m, apw_n and wbp_m and wpb_n. 
Investigate why. 
Investigate more on exactly how barriers work and the state machine of synchronization. we have some notes down as well. 
if 2 stages are the only needed thing, we may just use phase parity. not sure. 
We may also want to move to low precision stuff or persistent kernels, async stores are hard when the accumulator is float 32. 
