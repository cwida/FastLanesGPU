add_library(crystal STATIC ops/join.cu ops/project.cu)
target_include_directories(crystal PUBLIC ops)
target_include_directories(crystal PUBLIC ssb)
target_include_directories(crystal PUBLIC crystal)

add_executable(crystal_q11 ssb/q11.cu)
target_link_libraries(crystal_q11 crystal)

add_executable(crystal_q12 ssb/q12.cu)
target_link_libraries(crystal_q12 crystal)

add_executable(crystal_q13 ssb/q13.cu)
target_link_libraries(crystal_q13 crystal)

add_executable(crystal_q21 ssb/q21.cu)
target_link_libraries(crystal_q21 crystal)

add_executable(crystal_q22 ssb/q22.cu)
target_link_libraries(crystal_q22 crystal)

add_executable(crystal_q23 ssb/q23.cu)
target_link_libraries(crystal_q23 crystal)

add_executable(crystal_q31 ssb/q31.cu)
target_link_libraries(crystal_q31 crystal)

add_executable(crystal_q32 ssb/q32.cu)
target_link_libraries(crystal_q32 crystal)

add_executable(crystal_q33 ssb/q33.cu)
target_link_libraries(crystal_q33 crystal)

add_executable(crystal_q34 ssb/q34.cu)
target_link_libraries(crystal_q34 crystal)

add_executable(crystal_q41 ssb/q41.cu)
target_link_libraries(crystal_q41 crystal)

add_executable(crystal_q42 ssb/q42.cu)
target_link_libraries(crystal_q42 crystal)

add_executable(crystal_q43 ssb/q43.cu)
target_link_libraries(crystal_q43 crystal)