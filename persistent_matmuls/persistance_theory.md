Essentially, if A is row major and B is col major (which is optimal for our stuff, and is essentially non neg)
Then, for a persistent kernel we have (block_id, iter) -> tile_coord = (tile_m,tile_n)
for the same block id, but across different iterations, we don't care too much about Locality. 
WHY? simply because the all the blocks (all the sms) are gonna fetch A,B into L2 across K many times
so by the time one iteration of all blocks in paralell are done, lots of shit is gonna be evicted. 

the round robin assignment we are using (block_id, iter) -> tile_id = block_id + (num_sms*iter) 
essentially gives increasing tile id, first increasing by block_id, then increasing by iter 
so since spacial resure across the blocks in the same iteration matters more, this is perfect. 

if tile_id --row_major_interpret(lexical_map)--> (tile_m,tile_n) neighbouring sms acess close things from A 
if tile_id --col_major_interpret(co-lexical_map)--> (tile_m,tile_n) neighbouring sms access close things from B

suppose you had 8 sms, and A is row major (tile granular) and B is col major (tile granular)
then allow (block_id, iter) ---round_robin--> tile_coord --row_major---> (tile_m,tile_n) 
that at iteration zero, youd see the following map sm_id -> A_tile_offset(as in memory), B_tile_offset(as_in memory)

okay I wont actually write it out, my point is, it is often the case that 
