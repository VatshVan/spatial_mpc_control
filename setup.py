from setuptools import find_packages, setup

package_name = 'spatial_mpc_control'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
	data_files=[
	    ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
	    ('share/' + package_name, ['package.xml']),
	    ('share/' + package_name + '/launch', ['launch/sim.launch.py']),
	    ('share/' + package_name + '/urdf', ['urdf/spatial_platform.urdf.xacro']),
	],
    install_requires=['setuptools', 'torch', 'numpy', 'scipy', 'osqp', 'pandas'],
    zip_safe=True,
    maintainer='vatsh Van',
    maintainer_email='vatshvan.iitb@gmail.py',
    description='MIMO HyperRTIMPC spatial stabilization node',
    license='Proprietary',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'spatial_mpc_node = spatial_mpc_control.spatial_mpc_node:main'
        ],
    },
)
