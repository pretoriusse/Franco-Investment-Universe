import threading

# from .generate_p_dps_plots import main as p_dps_main
from .generate_p_ebita_ps_plots import main as p_ebita_main
from .generate_p_eps_ps_plots import main as p_eps_main
from .generate_p_fcf_ps_plots import main as fcf_main
from .generate_p_nav_plots import main as nav_main
from .generate_p_ntav_plots import main as ntav_main
from .generate_p_op_profit_ps_plots import main as op_profit_main
from .generate_p_sales_ps_plots import main as sales_main

p_ebita_main()
p_eps_main()
fcf_main()
nav_main()
ntav_main()
op_profit_main()
sales_main()
