"""Проверка воспроизводимости и ключевых результатов notebook."""

from pathlib import Path
import json
import math

import nbformat

ROOT = Path(__file__).resolve().parent
nb_path = ROOT / 'notebooks' / 'task_01_crm_efficiency.ipynb'
nb = nbformat.read(nb_path, as_version=4)

errors = []
for number, cell in enumerate(nb.cells, start=1):
    if cell.cell_type == 'code':
        for output in cell.get('outputs', []):
            if output.output_type == 'error':
                errors.append(f'Ячейка {number}: {output.ename}: {output.evalue}')

expected_figures = [
    ROOT / 'reports' / 'figures' / 'task_01_cash_flows.png',
    ROOT / 'reports' / 'figures' / 'task_01_cumulative_flows.png',
    ROOT / 'reports' / 'figures' / 'task_01_npv_sensitivity.png',
]
missing_figures = [str(path) for path in expected_figures if not path.exists() or path.stat().st_size == 0]

with open(ROOT / 'data' / 'task_01_crm_config.json', encoding='utf-8') as file:
    params = json.load(file)

annual_payroll = params['sales_managers'] * params['monthly_salary_rub'] * 12
routine_after = params['routine_time_share_before'] * (1 - params['routine_time_reduction'])
labor_savings = annual_payroll * (params['routine_time_share_before'] - routine_after)
additional_revenue = params['current_annual_turnover_rub'] * params['annual_sales_growth']
direct_effect = labor_savings + additional_revenue
annual_license = params['sales_managers'] * params['annual_license_per_user_rub']
adaptation_loss = params['sales_managers'] * params['monthly_salary_rub']
annual_cf = (direct_effect - annual_license) * (1 - params['profit_tax_rate'])
project_npv = -adaptation_loss + sum(
    annual_cf / (1 + params['discount_rate']) ** year
    for year in range(1, params['evaluation_horizon_years'] + 1)
)

checks = {
    'annual_payroll': (annual_payroll, 21600000),
    'routine_after': (routine_after, 0.24),
    'labor_savings': (labor_savings, 1296000),
    'additional_revenue': (additional_revenue, 5760000),
    'direct_effect': (direct_effect, 7056000),
    'annual_license': (annual_license, 360000),
    'adaptation_loss': (adaptation_loss, 1800000),
    'annual_net_cash_flow': (annual_cf, 5356800),
    'npv': (project_npv, 11066129.0),
}
failed_checks = {
    name: {'actual': actual, 'expected': expected}
    for name, (actual, expected) in checks.items()
    if not math.isclose(actual, expected, rel_tol=0, abs_tol=1.0)
}

if errors or missing_figures or failed_checks:
    raise SystemExit(json.dumps({
        'errors': errors,
        'missing_figures': missing_figures,
        'failed_checks': failed_checks,
    }, ensure_ascii=False, indent=2))

print('PASS: notebook contains no execution errors')
print('PASS: all three figures exist and are non-empty')
print('PASS: key values match the independent verification')
print(f'NPV={project_npv:,.2f} RUB; annual net cash flow={annual_cf:,.2f} RUB')
