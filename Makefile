################################################################################
# Makefile
#
#  * Shell
#  * Output Dirs
#  * Environment
#  * Datasets
#  * Tests
#  * Linters
#  * Phonies
#
################################################################################

################################################################################
# Settings
################################################################################

# Verify environment.sh
ifneq ($(PROJECT_NAME),llama-benchmarks)
$(error Environment not configured. Run `source environment.sh`)
endif


#-------------------------------------------------------------------------------
# Shell
#-------------------------------------------------------------------------------

# Bash
export SHELL := /bin/bash
.SHELLFLAGS := -e -u -o pipefail -c

# Colors - Supports colorized messages
COLOR_H1=\033[38;5;12m
COLOR_OK=\033[38;5;02m
COLOR_COMMENT=\033[38;5;08m
COLOR_RESET=\033[0m

# EXCLUDE_SRC - Source patterns to ignore

EXCLUDE_SRC := __pycache__ \
			   .egg-info \
			   .ipynb_checkpoints
EXCLUDE_SRC := $(subst $(eval ) ,|,$(EXCLUDE_SRC))


#-------------------------------------------------------------------------------
# Commands
#-------------------------------------------------------------------------------

RM := rm -rf


#-------------------------------------------------------------------------------
# Output Dirs
#-------------------------------------------------------------------------------

OUTPUT_DIRS :=

BUILD_DIR := .build
OUTPUT_DIRS := $(OUTPUT_DIRS) $(BUILD_DIR)


#-------------------------------------------------------------------------------
# Environment
#-------------------------------------------------------------------------------

VENV_ROOT := .venv
VENV := $(VENV_ROOT)/bin/activate


#-------------------------------------------------------------------------------
# Datasets
#-------------------------------------------------------------------------------

DATASETS_DIR := $(BUILD_DIR)/datasets
DATASETS :=

# Dataset: MMLU

MMLU_DATASET_SRC_URL := https://people.eecs.berkeley.edu/~hendrycks/data.tar
MMLU_DATASET_SRC := $(DATASETS_DIR)/mmlu.tar
MMLU_DATASET := $(DATASETS_DIR)/mmlu/README.txt
DATASETS := $(DATASETS) $(MMLU_DATASET)

#-------------------------------------------------------------------------------
# Tests
#-------------------------------------------------------------------------------

PYTEST_OPTS ?=


#-------------------------------------------------------------------------------
# Linters
#-------------------------------------------------------------------------------

RUFF_CHECK_OPTS ?= --preview
RUFF_FORMAT_OPTS ?= --preview


#-------------------------------------------------------------------------------
# Phonies
#-------------------------------------------------------------------------------

PHONIES :=


################################################################################
# Targets
################################################################################

all: venv datasets
	@echo
	@echo -e "$(COLOR_H1)# $(PROJECT_NAME)$(COLOR_RESET)"
	@echo
	@echo -e "$(COLOR_COMMENT)# Activate VENV$(COLOR_RESET)"
	@echo -e "source $(VENV)"
	@echo
	@echo -e "$(COLOR_COMMENT)# Deactivate VENV$(COLOR_RESET)"
	@echo -e "deactivate"
	@echo


#-------------------------------------------------------------------------------
# Output Dirs
#-------------------------------------------------------------------------------

$(BUILD_DIR):
	mkdir -p $@


#-------------------------------------------------------------------------------
# Environment
#-------------------------------------------------------------------------------

$(VENV): pyproject.toml
	uv sync

venv: $(VENV)
PHONIES := $(PHONIES) venv


#-------------------------------------------------------------------------------
# Datasets
#-------------------------------------------------------------------------------

$(DATASETS_DIR):
	mkdir -p $@

# Dataset: MMLU

$(MMLU_DATASET_SRC): | $(DATASETS_DIR)
	@echo
	@echo -e "$(COLOR_H1)# Dataset Source: MMLU$(COLOR_RESET)"
	@echo
	curl -L -o $@ $(MMLU_DATASET_SRC_URL)

$(MMLU_DATASET): $(MMLU_DATASET_SRC) | $(DATASETS_DIR)
	@echo
	@echo -e "$(COLOR_H1)# Dataset: MMLU$(COLOR_RESET)"
	@echo
	tar -C $(DATASETS_DIR) -xf $(MMLU_DATASET_SRC)
	mv $(DATASETS_DIR)/data $$(dirname $@)
	touch $@

datasets: $(DATASETS)
PHONIES := $(PHONIES) datasets

#-------------------------------------------------------------------------------
# Tests
#-------------------------------------------------------------------------------

tests: $(VENV) $(DATASETS)
	@echo
	@echo -e "$(COLOR_H1)# Tests$(COLOR_RESET)"
	@echo

	source $(VENV) && pytest $(PYTEST_OPTS) tests

coverage: $(VENV) $(DATASETS)
	@echo
	@echo -e "$(COLOR_H1)# Coverage$(COLOR_RESET)"
	@echo
	mkdir -p $$(dirname $(BUILD_DIR)/coverage)
	source $(VENV) && pytest $(PYTEST_OPTS) --cov=xformers --cov-report=html:$(BUILD_DIR)/coverage tests

PHONIES := $(PHONIES) tests coverage


#-------------------------------------------------------------------------------
# Linters
#-------------------------------------------------------------------------------

lint-fmt: venv
	source $(VENV) && \
	  ruff format $(RUFF_FORMAT_OPTS) && \
	  ruff check --fix $(RUFF_CHECK_OPTS) && \
	  make lint-style

lint-style: venv
	source $(VENV) && \
	  ruff check $(RUFF_CHECK_OPTS) && \
	  ruff format --check $(RUFF_FORMAT_OPTS)

PHONIES := $(PHONIES) lint-fmt lint-style


#-------------------------------------------------------------------------------
# Clean
#-------------------------------------------------------------------------------

clean-cache:
	find . -type d -name "__pycache__" -exec rm -rf {} +

clean-venv:
	$(RM) $(VENV_ROOT)

clean-build:
	$(RM) $(BUILD_DIR)

clean-datasets:
	$(RM) $(DATASETS_DIR)

clean: clean-cache clean-venv clean-build clean-datasets
PHONIES := $(PHONIES) clean-cache clean-venv clean-build clean-datasets clean


.PHONY: $(PHONIES)
