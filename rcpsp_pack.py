
#!/usr/bin/env python3
"""
RCPSP solver using linear search from upper bound down to lower bound.
This approach tries each makespan value sequentially to find the optimal solution.
No time limit per makespan test - only overall 1200s limit.
"""

"TODO: Tìm cách add trực tiếp bounds thành các constraint, ví dụ: Max(e1, e2,..., et) = 60"

from docplex.cp.model import *
import os
import csv
import time
from pathlib import Path
from google.cloud import storage
import os

# Thời gian tối đa cho mỗi instance
TIME_PER_INSTANCE = 1200


def solve_rcpsp_with_exact_makespan(data_file, target_makespan, lower_bound, time_remaining):
    """
    Test feasibility with exact makespan using both lower and upper bounds
    Returns True if feasible with exact target_makespan, False otherwise
    """
    try:
        # Read data file
        with open(data_file, 'r') as file:
            first_line = file.readline().split()
            NB_TASKS, NB_RESOURCES = int(first_line[0]), int(first_line[1])

            CAPACITIES = [int(v) for v in file.readline().split()]
            TASKS = [[int(v) for v in file.readline().split()] for i in range(NB_TASKS)]

        # Extract data
        DURATIONS = [TASKS[t][0] for t in range(NB_TASKS)]
        DEMANDS = [TASKS[t][1:NB_RESOURCES + 1] for t in range(NB_TASKS)]
        SUCCESSORS = [TASKS[t][NB_RESOURCES + 2:] for t in range(NB_TASKS)]

        # Create CP model
        mdl = CpoModel()

        # Create interval variables for tasks
        tasks = [interval_var(name=f'T{i + 1}', size=DURATIONS[i]) for i in range(NB_TASKS)]

        # Add precedence constraints
        for t in range(NB_TASKS):
            for s in SUCCESSORS[t]:
                if s > 0:  # Valid successor
                    mdl.add(end_before_start(tasks[t], tasks[s - 1]))

        # Add resource capacity constraints
        for r in range(NB_RESOURCES):
            resource_usage = [pulse(tasks[t], DEMANDS[t][r]) for t in range(NB_TASKS) if DEMANDS[t][r] > 0]
            if resource_usage:
                mdl.add(sum(resource_usage) <= CAPACITIES[r])

        # ADD CẢ LOWER VÀ UPPER BOUNDS để test chính xác target_makespan
        makespan = max(end_of(t) for t in tasks)
        mdl.add(makespan >= target_makespan)  # Lower bound constraint
        mdl.add(makespan <= target_makespan)  # Upper bound constraint
        # Tương đương với: mdl.add(makespan == target_makespan)

        # Solve with remaining time
        time_to_use = max(1, time_remaining)

        res = mdl.solve(
            TimeLimit=time_to_use,
            LogVerbosity="Quiet"
        )

        return res is not None and res.is_solution()

    except Exception as e:
        print(f"Error solving with exact makespan {target_makespan}: {str(e)}")
        return False


def solve_rcpsp_linear_search(data_file):
    """
    Solve RCPSP using linear search from upper bound down to lower bound
    Only overall TIME_PER_INSTANCE time limit - no limit per makespan test
    """
    start_time = time.time()

    try:
        # Read bounds from data file
        with open(data_file, 'r') as file:
            first_line = file.readline().split()
            NB_TASKS, NB_RESOURCES = int(first_line[0]), int(first_line[1])

            # Read bounds
            LOWER_BOUND = UPPER_BOUND = None
            if len(first_line) >= 4:
                LOWER_BOUND = int(first_line[2])
                UPPER_BOUND = int(first_line[3])
                print(f"Bounds from file: LB={LOWER_BOUND}, UB={UPPER_BOUND}")
            elif len(first_line) == 3:
                LOWER_BOUND = UPPER_BOUND = int(first_line[2])
                print(f"Single bound from file: {LOWER_BOUND}")
            else:
                print("No bounds specified in file")
                return (None, None, None, "infeasible", time.time() - start_time)

        if LOWER_BOUND is None or UPPER_BOUND is None:
            print("Invalid bounds")
            return (None, None, None, "infeasible", time.time() - start_time)

        # Linear search from upper bound down to lower bound
        print(f"Starting linear search from {UPPER_BOUND} down to {LOWER_BOUND}")
        print(f"Total time limit: {TIME_PER_INSTANCE}s")
        print(f"Strategy: Test exact makespan values with both lower and upper bounds")

        optimal_makespan = None
        attempts = 0
        timeout_occurred = False
        proven_infeasible_below = None  # Track the value proven infeasible

        for makespan in range(UPPER_BOUND, LOWER_BOUND - 1, -1):
            attempts += 1
            elapsed = time.time() - start_time
            time_remaining = TIME_PER_INSTANCE - elapsed

            # Check total time limit
            if time_remaining <= 0:
                print(f"Total time limit exceeded after {attempts} attempts")
                timeout_occurred = True
                break

            print(f"  Attempt {attempts}: Testing EXACT makespan = {makespan}")
            print(f"    Elapsed: {elapsed:.1f}s, Remaining: {time_remaining:.1f}s")

            # Test if this EXACT makespan is feasible
            attempt_start = time.time()
            is_feasible = solve_rcpsp_with_exact_makespan(data_file, makespan, LOWER_BOUND, time_remaining)
            attempt_time = time.time() - attempt_start

            print(f"    Attempt took: {attempt_time:.1f}s")

            if is_feasible:
                optimal_makespan = makespan
                print(f"  ✓ EXACT Makespan {makespan} is FEASIBLE")

                # Tìm được một giá trị khả thi, tiếp tục tìm giá trị nhỏ hơn
                continue
            else:
                print(f"  ✗ EXACT Makespan {makespan} is INFEASIBLE")

                # Ghi nhận giá trị đã chứng minh infeasible
                proven_infeasible_below = makespan

                # Nếu makespan hiện tại không khả thi, thì optimal_makespan
                # (nếu đã tìm được) là giá trị tốt nhất
                break

        solve_time = time.time() - start_time

        if optimal_makespan is not None:
            # LOGIC MỚI: Xác định status dựa trên việc có chứng minh được infeasible hay không
            if timeout_occurred:
                # Nếu timeout, chỉ có thể nói là feasible
                status = "feasible"
                print(f"✓ Found FEASIBLE solution (timeout): {optimal_makespan}")
            elif proven_infeasible_below is not None:
                # Nếu đã chứng minh được makespan nhỏ hơn là infeasible
                # thì optimal_makespan chính là OPTIMAL
                status = "optimal"
                print(f"✓ Found OPTIMAL solution: {optimal_makespan}")
                print(f"  Proven that makespan {proven_infeasible_below} is infeasible")
                print(f"  Therefore {optimal_makespan} is the smallest feasible makespan")
            elif optimal_makespan == LOWER_BOUND:
                # Nếu đạt được lower bound thì cũng là optimal
                status = "optimal"
                print(f"✓ Found OPTIMAL solution: {optimal_makespan} (matches lower bound)")
            else:
                # Trường hợp khác (không chắc chắn) thì chỉ là feasible
                status = "feasible"
                print(f"✓ Found FEASIBLE solution: {optimal_makespan}")

            print(f"Linear search completed: tested {attempts} exact values in {solve_time:.2f}s")
            return (LOWER_BOUND, UPPER_BOUND, optimal_makespan, status, solve_time)
        else:
            print(f"✗ No feasible solution found in range [{LOWER_BOUND}, {UPPER_BOUND}]")
            return (LOWER_BOUND, UPPER_BOUND, None, "infeasible", solve_time)

    except Exception as e:
        solve_time = time.time() - start_time
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return (None, None, None, "infeasible", solve_time)

def main():
    # Define directories
    data_dir = Path("data")
    result_dir = Path("result")
    output_file = result_dir / "pack_with_bound_1200s.csv"

    # Create result directory if it doesn't exist
    os.makedirs(result_dir, exist_ok=True)

    # Find all .data files in the data directory
    all_data_files = list(data_dir.glob("*.data"))
    if not all_data_files:
        print(f"Warning: No .data files found in {data_dir}")
        print(f"Current directory: {os.getcwd()}")
        print("Directory contents:")
        for item in os.listdir():
            print(f"  {item}")
        return

    # Sort files to ensure consistent order
    all_data_files.sort()

    # Process ALL files instead of a specific range
    data_files = all_data_files

    print(f"Found {len(all_data_files)} total .data files")
    print(f"Processing ALL {len(data_files)} files in the data directory")
    print(f"Using {TIME_PER_INSTANCE} seconds time limit per instance")
    print("Strategy: Linear search from upper bound down to lower bound")

    # Process files
    with open(output_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Updated CSV header as requested
        csv_writer.writerow(["File name", "LB", "UB", "Makespan", "Status", "Solve time (second)"])

        for i, data_file in enumerate(data_files, 1):
            file_name = data_file.name
            print(f"\n{'=' * 60}")
            print(f"[{i}/{len(data_files)}] Processing {file_name}")
            print(f"{'=' * 60}")

            try:
                lb, ub, makespan, status, solve_time = solve_rcpsp_linear_search(data_file)

                # Format values for CSV
                lb_str = str(lb) if lb is not None else "N/A"
                ub_str = str(ub) if ub is not None else "N/A"
                makespan_str = str(makespan) if makespan is not None else "N/A"

                csv_writer.writerow([
                    file_name,
                    lb_str,
                    ub_str,
                    makespan_str,
                    status,
                    f"{solve_time:.2f}"
                ])
                csvfile.flush()

                print(f"\nFINAL RESULT:")
                print(f"  File: {file_name}")
                print(f"  LB: {lb_str}")
                print(f"  UB: {ub_str}")
                print(f"  Makespan: {makespan_str}")
                print(f"  Status: {status}")
                print(f"  Time: {solve_time:.2f}s")

            except Exception as e:
                print(f"Error processing {file_name}: {str(e)}")
                csv_writer.writerow([file_name, "N/A", "N/A", "N/A", "infeasible", "0.00"])
                csvfile.flush()

    print(f"\n{'=' * 60}")
    print(f"ALL PROCESSING COMPLETE")
    print(f"Results saved to: {output_file}")
    print(f"{'=' * 60}")

    # Tên bucket mà bạn đã tạo
    bucket_name = "rcpsp-with-bounds-results-bucket"
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    local_path = "result/pack_with_bound_1200s.csv"
    blob_name = f"results/{os.path.basename(local_path)}"  # ví dụ "results/pack_with_bound_1200s.csv"

    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)
    print(f"Uploaded {local_path} to gs://{bucket_name}/{blob_name}")

if __name__ == "__main__":
    main()