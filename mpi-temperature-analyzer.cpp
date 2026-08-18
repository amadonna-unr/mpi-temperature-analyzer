#include <mpi.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstring>
#include <cstddef>
#include <cmath>
#include <omp.h>
#include <chrono>

struct Reading {
    std::string date;
    int tmax;
};

struct MPI_Record {
    char date[11];
    int tmax;
    int category;
};

void to_mpi_record(const Reading& src, MPI_Record& dst, int category = 0) {
    std::memset(dst.date, 0, sizeof(dst.date));
    std::strncpy(dst.date, src.date.c_str(), sizeof(dst.date) - 1);
    dst.tmax = src.tmax;
    dst.category = category;
}

Reading from_mpi_record(const MPI_Record& src) {
    Reading r;
    r.date = std::string(src.date);
    r.tmax = src.tmax;
    return r;
}

std::vector<Reading> load_readings(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("ERROR: daily_temps.csv not found in the current directory.");
    }

    std::vector<Reading> readings;
    std::string line;

    std::getline(file, line);
    std::getline(file, line);

    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;
        }

        std::stringstream ss(line);
        std::string field;
        std::vector<std::string> cols;

        while (std::getline(ss, field, ',')) {
            cols.push_back(field);
        }

        if (cols.size() < 3) {
            continue;
        }

        Reading r;
        r.date = cols[0];
        r.tmax = std::stoi(cols[2]);

        if (!r.date.empty()) {
            readings.push_back(r);
        }
    }

    return readings;
}

void print_group(const std::string& label, const std::vector<MPI_Record>& items) {
    std::cout << label << "\n";
    if (items.empty()) {
        std::cout << "  (none)\n";
        return;
    }

    for (const auto& item : items) {
        std::cout << "  " << item.date << ": " << item.tmax << "\n";
    }
}

int main(int argc, char* argv[]) {
    
    auto startTime = std::chrono::high_resolution_clock::now();

    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<Reading> all_readings;
    if (rank == 0) {
        try {
            all_readings = load_readings("daily_temps.csv");
        } catch (const std::exception& e) {
            std::cerr << e.what() << '\n';
        }
    }

    int total_count = static_cast<int>(all_readings.size());
    MPI_Bcast(&total_count, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (total_count <= 0) {
        if (rank == 0) {
            std::cerr << "No valid readings were loaded.\n";
        }
        MPI_Finalize();
        return 0;
    }

    std::vector<int> counts(size, 0);
    std::vector<int> displs(size, 0);

    int base = total_count / size;
    int remainder = total_count % size;

    for (int i = 0; i < size; ++i) {
        counts[i] = base + (i < remainder ? 1 : 0);
    }
    for (int i = 1; i < size; ++i) {
        displs[i] = displs[i - 1] + counts[i - 1];
    }

    int local_count = counts[rank];
    std::vector<MPI_Record> local(local_count);

    MPI_Datatype mpi_record_type;
    int block_lengths[3] = {11, 1, 1};
    MPI_Aint offsets[3] = {
        offsetof(MPI_Record, date),
        offsetof(MPI_Record, tmax),
        offsetof(MPI_Record, category)
    };
    MPI_Datatype types[3] = {MPI_CHAR, MPI_INT, MPI_INT};

    MPI_Type_create_struct(3, block_lengths, offsets, types, &mpi_record_type);
    MPI_Type_commit(&mpi_record_type);

    if (rank == 0) {
        std::vector<MPI_Record> sendbuf(total_count);
        for (int i = 0; i < total_count; ++i) {
            to_mpi_record(all_readings[i], sendbuf[i]);
        }

        MPI_Scatterv(sendbuf.data(), counts.data(), displs.data(), mpi_record_type,
                     local.data(), local_count, mpi_record_type,
                     0, MPI_COMM_WORLD);
    } else {
        MPI_Scatterv(nullptr, nullptr, nullptr, mpi_record_type,
                     local.data(), local_count, mpi_record_type,
                     0, MPI_COMM_WORLD);
    }

    double local_sum = 0.0;
#pragma omp parallel for reduction(+:local_sum)
    for (int i = 0; i < static_cast<int>(local.size()); ++i) {
        local_sum += static_cast<double>(local[i].tmax);
    }

    double global_sum = 0.0;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    int global_count = 0;
    MPI_Allreduce(&local_count, &global_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    double mean = 0.0;
    if (rank == 0) {
        mean = global_sum / static_cast<double>(global_count);
    }
    MPI_Bcast(&mean, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(local.size()); ++i) {
        double diff = std::fabs(static_cast<double>(local[i].tmax) - mean);
        if (diff <= 0.5) {
            local[i].category = 0;
        } else if (local[i].tmax > mean) {
            local[i].category = 1;
        } else {
            local[i].category = -1;
        }
    }

    std::vector<MPI_Record> gathered(total_count);
    MPI_Gatherv(local.data(), local_count, mpi_record_type,
                gathered.data(), counts.data(), displs.data(), mpi_record_type,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::vector<MPI_Record> over;
        std::vector<MPI_Record> under;
        std::vector<MPI_Record> at_average;

        for (const auto& item : gathered) {
            if (item.category == 1) {
                over.push_back(item);
            } else if (item.category == -1) {
                under.push_back(item);
            } else {
                at_average.push_back(item);
            }
        }

        std::cout << "Average TMAX across all days: " << mean << "\n\n";
        print_group("OVER AVERAGE:", over);
        print_group("UNDER AVERAGE:", under);
        print_group("AT AVERAGE (" + std::to_string(at_average.size()) + "):", at_average);
    }

    MPI_Type_free(&mpi_record_type);
    MPI_Finalize();

    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = endTime - startTime;

    std::cout << elapsed.count() << std::endl;

    return 0;
}