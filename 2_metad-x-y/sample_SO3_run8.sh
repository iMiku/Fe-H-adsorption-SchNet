#!/bin/bash

# Function to generate a random axis-angle pair
generate_axis_angle() {
  # Generate random axis components, normalized to unit length
  x=$(echo "scale=3; $RANDOM/32767 - 0.5" | bc)
  y=$(echo "scale=3; $RANDOM/32767 - 0.5" | bc)
  z=$(echo "scale=3; $RANDOM/32767 - 0.5" | bc)
  
  # Compute magnitude to normalize the vector
  norm=$(echo "scale=5; sqrt($x^2 + $y^2 + $z^2)" | bc)

  # Normalize the axis components to get a unit vector
  unit_x=$(echo "scale=3; $x / $norm" | bc)
  unit_y=$(echo "scale=3; $y / $norm" | bc)
  unit_z=$(echo "scale=3; $z / $norm" | bc)

  # Add zero before numbers that start with a decimal point
  unit_x=$(printf "%.3f" $unit_x)
  unit_y=$(printf "%.3f" $unit_y)
  unit_z=$(printf "%.3f" $unit_z)

  # Generate random angle in degrees (integer between 0 and 360)
  angle=$((RANDOM % 361))

  # Save axis and angle to variables
  AXIS_X=$unit_x
  AXIS_Y=$unit_y
  AXIS_Z=$unit_z
  ANGLE=$angle
}

lmp_pre="/home/pc/lammps/lmp_mpi"
lmp_suf1="-screen none -in iron_cube_gen.in"
lmp_suf2="-screen none -in iron_add_h2.in"
lmp_suf4="-screen none -in np_iron_h2_xy.in"

for i in {1..8}; do
  # Call the function to generate a random axis-angle pair
  generate_axis_angle
  r1vx=$AXIS_X
  r1vy=$AXIS_Y
  r1vz=$AXIS_Z
  r1ad=$angle
  #generate_axis_angle
  #r2vx=$AXIS_X
  #r2vy=$AXIS_Y
  #r2vz=$AXIS_Z
  #r2ad=$angle
  lmp_para1="-var rot1_vec_x ${r1vx} -var rot1_vec_y ${r1vy} -var rot1_vec_z ${r1vz} -var rot1_deg ${r1ad}"
  #lmp_para2="-var rot2_vec_x ${r2vx} -var rot2_vec_y ${r2vy} -var rot2_vec_z ${r2vz} -var rot2_deg ${r2ad}"
  out_suf=">nohup.out${i} & disown"
  cmd1="${lmp_pre} ${lmp_para1} ${lmp_suf1}"
  cmd2="${lmp_pre} ${lmp_para1} ${lmp_suf2}"
  data_name="box4r1x${r1vx}r1y${r1vy}r1z${r1vz}r1d${r1ad}"
  cmd3="python ovito_set_image_zero.py ${data_name}.equ.min.data"
  cmd4="${lmp_pre} ${lmp_para1} ${lmp_suf4}"
  cmd_1234="nohup bash -c \"${cmd1} && ${cmd2} && ${cmd3} && ${cmd4}\" ${out_suf}"
  echo ${cmd_1234} 
done
