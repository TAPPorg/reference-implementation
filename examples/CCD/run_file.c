#define _GNU_SOURCE // for cpu_set_t and the CPU_* macros
#include "run_file.h"

#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LINE 1024
#define MAX_FIELDS 16

struct parser
{
    const char* path;
    int line;           // number of the current line, starting at 1
    int term_count;
    int* step_lines;    // per term the line it is run on, or 0 if it is not run yet
    struct run_file* run;
};

// Prints a problem with the current line. Returns 1, to be passed on as the error.
static int error_at(const struct parser* p, const char* format, ...)
{
    fprintf(stderr, "%s:%d: ", p->path, p->line);
    va_list args;
    va_start(args, format);
    vfprintf(stderr, format, args);
    va_end(args);
    fputc('\n', stderr);
    return 1;
}

// Splits a line at whitespace, in place. Returns the number of fields, at most MAX_FIELDS + 1.
static int split_fields(char* line, char** fields)
{
    int count = 0;
    char* rest;
    for (char* field = strtok_r(line, " \t\r\n", &rest); field != NULL && count <= MAX_FIELDS;
         field = strtok_r(NULL, " \t\r\n", &rest))
    {
        fields[count++] = field;
    }
    return count;
}

// Reads a whole string of digits as an int. Returns 0 on success.
static int parse_int(const char* text, int* value)
{
    if (text[0] < '0' || text[0] > '9') return 1;
    char* end;
    errno = 0;
    long number = strtol(text, &end, 10);
    if (*end != '\0' || errno != 0 || number > INT_MAX) return 1;
    *value = (int)number;
    return 0;
}

// Reads "<n>" or "<first>-<last>", in place. If prefix is not 0, every number has to start with it, like the c of
// c1-c31. Returns 0 on success.
static int parse_range(char* text, char prefix, int* first, int* last)
{
    char* first_text = text;
    char* last_text = text;
    char* dash = strchr(text, '-');
    if (dash != NULL)
    {
        *dash = '\0';
        last_text = dash + 1;
    }
    if (prefix != 0)
    {
        if (*first_text != prefix || *last_text != prefix) return 1;
        first_text++;
        last_text++;
    }
    if (parse_int(first_text, first) != 0 || parse_int(last_text, last) != 0) return 1;
    return *first > *last;
}

// Reads a list of CPUs like 1,3,6-7,12, in place. Returns 0 on success.
static int parse_cpus(char* text, cpu_set_t* cpus)
{
    CPU_ZERO(cpus);
    char* rest;
    for (char* item = strtok_r(text, ",", &rest); item != NULL; item = strtok_r(NULL, ",", &rest))
    {
        int first, last;
        if (parse_range(item, 0, &first, &last) != 0 || last >= CPU_SETSIZE) return 1;
        for (int cpu = first; cpu <= last; cpu++) CPU_SET(cpu, cpus);
    }
    return CPU_COUNT(cpus) == 0;
}

// Position of the backend with the name, or -1
static int find_backend(const struct run_file* run, const char* name)
{
    for (int i = 0; i < run->backend_count; i++)
    {
        if (strcmp(run->backends[i].name, name) == 0) return i;
    }
    return -1;
}

// Position of the executor with the name, or -1
static int find_executor(const struct run_file* run, const char* name)
{
    for (int i = 0; i < run->executor_count; i++)
    {
        if (strcmp(run->executors[i].name, name) == 0) return i;
    }
    return -1;
}

// backend <name> <path>
static int parse_backend(struct parser* p, char** fields, int count)
{
    struct run_file* run = p->run;
    if (count != 3) return error_at(p, "expected: backend <name> <path>");
    if (strlen(fields[1]) >= RUN_NAME_MAX) return error_at(p, "the name %s is too long", fields[1]);
    if (strlen(fields[2]) >= PATH_MAX) return error_at(p, "the path of backend %s is too long", fields[1]);
    if (find_backend(run, fields[1]) >= 0) return error_at(p, "there already is a backend %s", fields[1]);

    struct run_backend* backends = realloc(run->backends, (size_t)(run->backend_count + 1) * sizeof(*backends));
    if (backends == NULL) return error_at(p, "out of memory");
    run->backends = backends;

    struct run_backend* backend = &backends[run->backend_count++];
    strcpy(backend->name, fields[1]);
    strcpy(backend->path, fields[2]);
    return 0;
}

// executor <name> <backend> [<attribute>=<value> ...]
static int parse_executor(struct parser* p, char** fields, int count)
{
    struct run_file* run = p->run;
    if (count < 3) return error_at(p, "expected: executor <name> <backend> [<attribute>=<value> ...]");
    if (strlen(fields[1]) >= RUN_NAME_MAX) return error_at(p, "the name %s is too long", fields[1]);
    if (find_executor(run, fields[1]) >= 0) return error_at(p, "there already is an executor %s", fields[1]);

    struct run_executor executor = {0};
    strcpy(executor.name, fields[1]);
    executor.backend = find_backend(run, fields[2]);
    if (executor.backend < 0) return error_at(p, "there is no backend %s", fields[2]);

    for (int i = 3; i < count; i++)
    {
        char* value = strchr(fields[i], '=');
        if (value == NULL) return error_at(p, "expected <attribute>=<value> instead of %s", fields[i]);
        *value++ = '\0';
        if (strcmp(fields[i], "cpus") == 0)
        {
            if (parse_cpus(value, &executor.cpus) != 0) return error_at(p, "cpus is not a list of CPUs like 0-3,6");
            executor.has_cpus = 1;
        }
        else if (strcmp(fields[i], "threads") == 0)
        {
            if (parse_int(value, &executor.threads) != 0 || executor.threads == 0)
            {
                return error_at(p, "threads is not a positive number");
            }
        }
        else
        {
            return error_at(p, "there is no attribute %s", fields[i]);
        }
    }

    struct run_executor* executors = realloc(run->executors, (size_t)(run->executor_count + 1) * sizeof(*executors));
    if (executors == NULL) return error_at(p, "out of memory");
    run->executors = executors;
    executors[run->executor_count++] = executor;
    return 0;
}

// run <term or range of terms> <executor>
static int parse_run(struct parser* p, char** fields, int count)
{
    struct run_file* run = p->run;
    if (count != 3) return error_at(p, "expected: run <term or range of terms> <executor>");

    int first, last;
    if (parse_range(fields[1], 'c', &first, &last) != 0) return error_at(p, "expected a term like c1 or c1-c31");
    if (first < 1 || last > p->term_count) return error_at(p, "the terms are c1 to c%d", p->term_count);

    int executor = find_executor(run, fields[2]);
    if (executor < 0) return error_at(p, "there is no executor %s", fields[2]);

    for (int term = first - 1; term < last; term++)
    {
        if (p->step_lines[term] != 0)
        {
            return error_at(p, "term c%d is already run on line %d", term + 1, p->step_lines[term]);
        }
        p->step_lines[term] = p->line;
        run->steps[run->step_count].term = term;
        run->steps[run->step_count].executor = executor;
        run->step_count++;
    }
    return 0;
}

static int parse_line(struct parser* p, char* line)
{
    char* fields[MAX_FIELDS + 1];
    int count = split_fields(line, fields);
    if (count == 0 || fields[0][0] == '#') return 0;
    if (count > MAX_FIELDS) return error_at(p, "more than %d fields", MAX_FIELDS);

    if (strcmp(fields[0], "backend") == 0) return parse_backend(p, fields, count);
    if (strcmp(fields[0], "executor") == 0) return parse_executor(p, fields, count);
    if (strcmp(fields[0], "run") == 0) return parse_run(p, fields, count);
    return error_at(p, "expected backend, executor or run instead of %s", fields[0]);
}

int read_run_file(const char* path, int term_count, struct run_file* run)
{
    memset(run, 0, sizeof(*run));

    FILE* file = fopen(path, "r");
    if (file == NULL)
    {
        fprintf(stderr, "%s: %s\n", path, strerror(errno));
        return 1;
    }

    struct parser p = {.path = path, .term_count = term_count, .run = run};
    // One element more than needed, so that there is something to allocate without terms
    p.step_lines = calloc((size_t)term_count + 1, sizeof(*p.step_lines));
    run->steps = calloc((size_t)term_count + 1, sizeof(*run->steps));
    int error = 0;
    if (p.step_lines == NULL || run->steps == NULL)
    {
        fprintf(stderr, "%s: out of memory\n", path);
        error = 1;
    }

    char line[MAX_LINE];
    while (error == 0 && fgets(line, sizeof(line), file) != NULL)
    {
        p.line++;
        if (strchr(line, '\n') == NULL && !feof(file))
        {
            error = error_at(&p, "the line is longer than %d characters", MAX_LINE - 2);
            break;
        }
        error = parse_line(&p, line);
    }
    fclose(file);

    for (int term = 0; error == 0 && term < term_count; term++)
    {
        if (p.step_lines[term] == 0)
        {
            fprintf(stderr, "%s: term c%d is not run\n", path, term + 1);
            error = 1;
        }
    }

    free(p.step_lines);
    if (error != 0) free_run_file(run);
    return error;
}

void free_run_file(struct run_file* run)
{
    free(run->backends);
    free(run->executors);
    free(run->steps);
    memset(run, 0, sizeof(*run));
}
