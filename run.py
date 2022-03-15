from pathlib import Path
import sys
import os

biopal_path = Path("/data/largeHome/guli_ro/python/biopal/BioPAL")

sys.path.append(str(biopal_path))
os.chdir(biopal_path)

# test = 'biopal' 'apps'
test = "biopal"

input_file_xml_path = r"/data/largeHome/guli_ro/python/biopal/BioPAL/inputs/Input_File.xml"
configuration_file = r"/biopal/conf/Configuration_File.xml"

if test == "biopal":

    from biopal.__main__ import biomassL2_processor_run

    conf_folder = biopal_path.joinpath("biopal", "conf")

    biomassL2_processor_run(input_file_xml_path, conf_folder)


elif test == "apps":

    from biopal.dataset_query.dataset_query import dataset_query
    from biopal.fh.main_FH import StackBasedProcessingFH, CoreProcessingFH

    dataset_query_obj = dataset_query()
    input_file_with_stack_base_proc = dataset_query_obj.run(input_file_xml_path)

    # Main APP #1: Stack Based Processing
    stack_based_processing_obj = StackBasedProcessingFH(configuration_file)

    # Run Main APP #1: Stack Based Processing
    input_file_updated = stack_based_processing_obj.run(input_file_with_stack_base_proc)

    # Main APP #2: Core Processing
    fh_processing_obj = CoreProcessingFH(configuration_file)

    # Run Main APP #2: AGB Core Processing
    fh_processing_obj.run(input_file_updated)


    #ss=ss.ReadAsArray()
    #ss=gdal.Open("/data/largeHome/guli_ro/python/biopal/BioPAL/output_old/BIOMASS_L2_20210922T153054/FH/Products/global_FH/EQUI7_AF050M/E045N048T3/FH.tif")

    # import matplotlib.pyplot as plt
    # plt.imshow(MBMP_correlation[0,1,:,:])
