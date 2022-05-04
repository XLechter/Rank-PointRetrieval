import get_regist_score

file_cur = 'inhouse_datasets/business_run2/pointcloud_25m_25/15085034089234540.bin'
files_match = {'file': 'inhouse_datasets/business_run1/pointcloud_25m_25/15085021101060450.bin', 'id2': 8, 'gt': True, 'cur': 'm0n1i8'}, {'file': 'inhouse_datasets/business_run1/pointcloud_25m_25/15085023377674230.bin', 'id2': 58, 'gt': False, 'cur': 'm0n1i8'}, {'file': 'inhouse_datasets/business_run1/pointcloud_25m_25/15085022577800300.bin', 'id2': 43, 'gt': False, 'cur': 'm0n1i8'}, {'file': 'inhouse_datasets/business_run1/pointcloud_25m_25/15085025241993290.bin', 'id2': 103, 'gt': False, 'cur': 'm0n1i8'}, {'file': 'inhouse_datasets/business_run1/pointcloud_25m_25/15085021065005400.bin', 'id2': 7, 'gt': False, 'cur': 'm0n1i8'}, {'file': 'inhouse_datasets/business_run1/pointcloud_25m_25/15085027759950070.bin', 'id2': 168, 'gt': False, 'cur': 'm0n1i8'}, {'file': 'inhouse_datasets/business_run1/pointcloud_25m_25/15085026265704740.bin', 'id2': 127, 'gt': False, 'cur': 'm0n1i8'}, {'file': 'inhouse_datasets/business_run1/pointcloud_25m_25/15085021659962610.bin', 'id2': 22, 'gt': False, 'cur': 'm0n1i8'}, {'file': 'inhouse_datasets/business_run1/pointcloud_25m_25/15085028993806460.bin', 'id2': 191, 'gt': False, 'cur': 'm0n1i8'}, {'file': 'inhouse_datasets/business_run1/pointcloud_25m_25/15085026323771260.bin', 'id2': 129, 'gt': False, 'cur': 'm0n1i8'}, {'file': 'inhouse_datasets/business_run1/pointcloud_25m_25/15085024609109700.bin', 'id2': 87, 'gt': False, 'cur': 'm0n1i8'}, {'file': 'inhouse_datasets/business_run1/pointcloud_25m_25/15085027729903420.bin', 'id2': 167, 'gt': False, 'cur': 'm0n1i8'}, {'file': 'inhouse_datasets/business_run1/pointcloud_25m_25/15085022743091570.bin', 'id2': 47, 'gt': False, 'cur': 'm0n1i8'}, {'file': 'inhouse_datasets/business_run1/pointcloud_25m_25/15085026750026530.bin', 'id2': 143, 'gt': False, 'cur': 'm0n1i8'}, {'file': 'inhouse_datasets/business_run1/pointcloud_25m_25/15085026717040560.bin', 'id2': 142, 'gt': False, 'cur': 'm0n1i8'}, {'file': 'inhouse_datasets/business_run1/pointcloud_25m_25/15085021030009840.bin', 'id2': 6, 'gt': False, 'cur': 'm0n1i8'}, {'file': 'inhouse_datasets/business_run1/pointcloud_25m_25/15085023305666150.bin', 'id2': 56, 'gt': False, 'cur': 'm0n1i8'}, {'file': 'inhouse_datasets/business_run1/pointcloud_25m_25/15085021261483670.bin', 'id2': 12, 'gt': False, 'cur': 'm0n1i8'}, {'file': 'inhouse_datasets/business_run1/pointcloud_25m_25/15085022709132590.bin', 'id2': 46, 'gt': False, 'cur': 'm0n1i8'}, {'file': 'inhouse_datasets/business_run1/pointcloud_25m_25/15085027788944300.bin', 'id2': 169, 'gt': False, 'cur': 'm0n1i8'}, {'file': 'inhouse_datasets/business_run1/pointcloud_25m_25/15085024454877670.bin', 'id2': 83, 'gt': False, 'cur': 'm0n1i8'}, {'file': 'inhouse_datasets/business_run1/pointcloud_25m_25/15085025108514820.bin', 'id2': 99, 'gt': False, 'cur': 'm0n1i8'}, {'file': 'inhouse_datasets/business_run1/pointcloud_25m_25/15085027314467560.bin', 'id2': 154, 'gt': False, 'cur': 'm0n1i8'}, {'file': 'inhouse_datasets/business_run1/pointcloud_25m_25/15085028801686210.bin', 'id2': 186, 'gt': False, 'cur': 'm0n1i8'}, {'file': 'inhouse_datasets/business_run1/pointcloud_25m_25/15085028963763260.bin', 'id2': 190, 'gt': False, 'cur': 'm0n1i8'}
rs = get_regist_score.evaluate_matchs(file_cur, files_match, log=False)