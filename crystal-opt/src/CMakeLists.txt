add_library(crystal_opt STATIC ops/join.cu ops/project.cu)
target_include_directories(crystal_opt PUBLIC ops)
target_include_directories(crystal_opt PUBLIC ssb)
target_include_directories(crystal_opt PUBLIC crystal)

add_executable(crystal_opt_q11 ssb/q11.cu)
target_link_libraries(crystal_opt_q11 crystal_opt)

add_executable(crystal_opt_q12 ssb/q12.cu)
target_link_libraries(crystal_opt_q12 crystal_opt)

add_executable(crystal_opt_q13 ssb/q13.cu)
target_link_libraries(crystal_opt_q13 crystal_opt)

add_executable(crystal_opt_q21 ssb/q21.cu)
target_link_libraries(crystal_opt_q21 crystal_opt)

add_executable(crystal_opt_q22 ssb/q22.cu)
target_link_libraries(crystal_opt_q22 crystal_opt)

add_executable(crystal_opt_q23 ssb/q23.cu)
target_link_libraries(crystal_opt_q23 crystal_opt)

add_executable(crystal_opt_q31 ssb/q31.cu)
target_link_libraries(crystal_opt_q31 crystal_opt)

add_executable(crystal_opt_q32 ssb/q32.cu)
target_link_libraries(crystal_opt_q32 crystal_opt)

add_executable(crystal_opt_q33 ssb/q33.cu)
target_link_libraries(crystal_opt_q33 crystal_opt)

add_executable(crystal_opt_q34 ssb/q34.cu)
target_link_libraries(crystal_opt_q34 crystal_opt)

add_executable(crystal_opt_q41 ssb/q41.cu)
target_link_libraries(crystal_opt_q41 crystal_opt)

add_executable(crystal_opt_q42 ssb/q42.cu)
target_link_libraries(crystal_opt_q42 crystal_opt)

add_executable(crystal_opt_q43 ssb/q43.cu)
target_link_libraries(crystal_opt_q43 crystal_opt)