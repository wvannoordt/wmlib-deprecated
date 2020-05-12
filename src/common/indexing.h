#ifndef INDEXING_H
#define INDEXING_H

#define WM_NRAY globaldata.settings.ray_point_count
#define WM_NUMWALLPOINTS globaldata.instance.num_wall_points
#define WM_DIM (2+iins3d)
#define WM_DIM2 (2+iins3d)*(2+iins3d)
#define BUFFERVARIDX(my_buf, my_index) globaldata.buffer.solution.my_buf.base[(my_index)*WM_NUMWALLPOINTS + buffer_index]
#define BUFFERVARIDXPTR(my_buf) globaldata.buffer.solution.my_buf.base + buffer_index
#define INIDX(my_buf) globaldata.buffer.in.my_buf.base[buffer_index]
#define INIDX_T1D(my_buf, my_idx) globaldata.buffer.in.my_buf.base[(buffer_index)*WM_DIM2 + my_idx]
#define OUTIDX(my_buf) globaldata.buffer.out.my_buf.base[buffer_index]
#define OUTIDXPTR(my_buf) globaldata.buffer.out.my_buf.base + buffer_index
#define OUTIDX_T1D(my_buf, my_idx) globaldata.buffer.out.my_buf.base[(buffer_index)*WM_DIM2 + my_idx]
#define OUTIDX_V(my_buf, my_idx) globaldata.buffer.out.my_buf.base[(buffer_index)*WM_DIM + my_idx]
#define GETLOCALVARS(my_buf,my_idx,my_array) *(my_array+0)=BUFFERVARIDX(my_buf,my_idx-1);*(my_array+1)=BUFFERVARIDX(my_buf,my_idx);*(my_array+2)=BUFFERVARIDX(my_buf,my_idx+1);
#define GETLOCALVARS_F(my_buf,my_array) *(my_array+0)=INIDX(my_buf);*(my_array+1)=INIDX(my_buf);*(my_array+2)=INIDX(my_buf);

#endif
