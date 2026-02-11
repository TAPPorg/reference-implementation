add_library(tapp-api INTERFACE)
set_property(TARGET tapp-api PROPERTY EXPORT_NAME api)
add_library(tapp::api ALIAS tapp-api)

target_include_directories(tapp-api INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${TAPP_INSTALL_INCLUDEDIR}>
)

install(TARGETS tapp-api EXPORT tapp
        COMPONENT api)
install(DIRECTORY include/
        COMPONENT api
        DESTINATION "${TAPP_INSTALL_INCLUDEDIR}")
