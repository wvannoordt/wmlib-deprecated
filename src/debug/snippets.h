for (int i = 0; i < source_count; i++)
{
    std::cout << *(transfer_endpoints[i]) << " + " << transfer_endpoint_offsets[i] << " s" << transfer_stride[i];
    std::cout << " <<<--- ";
    std::cout << *(transfer_sources[i]) << " + " << transfer_source_offsets[i] << " s" << transfer_stride[i];
    std::cout << ((mem_copy_dirs[i] == cudaMemcpyDeviceToHost) ? " cudaMemcpyDeviceToHost " : " cudaMemcpyHostToDevice ");
    std::cout << "(" << (int)transfer_protocol << ") - var "  << transfer_endpoint_names[i] << " << " << transfer_source_names[i];
    std::cout << " " << source_count << ", " << endpoint_count << std::endl;
}
