
# Load necessary libraries
library(ggplot2)
library(dplyr)
library(reshape2)
library(gridExtra)

# Load the data
# aggregated_df <- read.csv("model_performance.csv")

# 1. Calculate average metrics by Prompt Type and Model
average_metrics <- aggregated_df %>%
  group_by(Task, Model) %>%
  summarise(`GPU Power (W)` = mean(`GPU Power (W)`, na.rm = TRUE)) %>%
  ungroup()

# 2. Line Plot for GPU Utilization Over 30 Runs
ggplot(aggregated_df, aes(x = Repetition, y = `GPU Utilization (%)`, color = Task, linetype = Model, group = interaction(Task, Model))) +
  geom_line(size = 1) +
  geom_point(size = 2) +
  ggtitle("GPU Utilization Over 30 Runs for Each Task and Model") +
  xlab("Repetition (Run #)") +
  ylab("GPU Utilization (%)") +
  scale_color_brewer(palette = "Set1", name = "Task") +
  theme_minimal() +
  theme(
    legend.position = "top",
    axis.text.x = element_text(angle = 0, vjust = 0.5, hjust = 0.5)
  )

# 3. Scatter Plot for GPU Power vs. GPU Utilization
ggplot(aggregated_df, aes(x = `GPU Utilization (%)`, y = `GPU Power (W)`, color = Task, shape = Model)) +
  geom_point(size = 3, alpha = 0.7) +
  ggtitle("GPU Power vs. GPU Utilization Over 30 Runs for Each Task and Model") +
  xlab("GPU Utilization (%)") +
  ylab("GPU Power (W)") +
  scale_color_brewer(palette = "Set1", name = "Task") +
  theme_minimal() +
  theme(
    legend.position = "top",
    axis.text.x = element_text(angle = 0, vjust = 0.5, hjust = 0.5)
  )

# 4. Scatter Plot for GPU Utilization vs. Inference Time
ggplot(aggregated_df, aes(x = `GPU Utilization (%)`, y = `Duration (s)`, color = Model)) +
  geom_point(size = 3, alpha = 0.7) +
  ggtitle("GPU Utilization vs. Inference Time") +
  xlab("GPU Utilization (%)") +
  ylab("Inference Time (s)") +
  scale_color_brewer(palette = "Set1", name = "Model") +
  theme_minimal() +
  theme(
    legend.position = "top",
    axis.text.x = element_text(angle = 0, vjust = 0.5, hjust = 0.5)
  )

# 5. Scatter Plot for CPU Utilization vs. Inference Time
ggplot(aggregated_df, aes(x = `CPU Utilization (%)`, y = `Duration (s)`, color = Model)) +
  geom_point(size = 3, alpha = 0.7) +
  ggtitle("CPU Utilization vs. Inference Time") +
  xlab("CPU Utilization (%)") +
  ylab("Inference Time (s)") +
  scale_color_brewer(palette = "Set1", name = "Model") +
  theme_minimal() +
  theme(
    legend.position = "top",
    axis.text.x = element_text(angle = 0, vjust = 0.5, hjust = 0.5)
  )

# 6. Correlation Heatmap Between Inference Time and GPU Power
correlation_matrix <- aggregated_df %>%
  select(`Duration (s)`, `GPU Power (W)`) %>%
  cor(use = "complete.obs")
correlation_melted <- melt(correlation_matrix)

ggplot(correlation_melted, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0, 
                       limit = c(-1, 1), space = "Lab", name = "Correlation") +
  geom_text(aes(label = round(value, 2)), color = "black", size = 4) +
  ggtitle("Correlation Between Inference Time and GPU Power") +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1),
    axis.title.x = element_blank(),
    axis.title.y = element_blank()
  )

# 7. Bar Plot for Average GPU Power by Prompt Type and Model
ggplot(average_metrics, aes(x = Task, y = `GPU Power (W)`, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge") +
  ggtitle("Average GPU Power Consumption by Prompt Type and Model") +
  xlab("Prompt Type") +
  ylab("GPU Power (W)") +
  scale_fill_brewer(palette = "Set1", name = "Model") +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "top"
  )

# 8. Line Plot for Inference Duration Across Repetitions
ggplot(aggregated_df, aes(x = Repetition, y = `Duration (s)`, color = Task, linetype = Model, group = interaction(Task, Model))) +
  geom_line(size = 1) +
  geom_point(size = 2) +
  ggtitle("Inference Duration Across Repetitions for Each Task and Model") +
  xlab("Repetition (Run number)") +
  ylab("Inference Duration (Seconds)") +
  scale_color_brewer(palette = "Set1", name = "Task") +
  theme_minimal() +
  theme(
    legend.position = "top",
    axis.text.x = element_text(angle = 0, vjust = 0.5, hjust = 0.5)
  )

# 9. Statistical Analysis

# Check Normality using Shapiro-Wilk Test
shapiro_test <- shapiro.test(aggregated_df$energy_consumption)
cat("Shapiro-Wilk Test p-value:", shapiro_test$p.value, "\n")

# Q-Q Plot for Energy Consumption
qqnorm(aggregated_df$energy_consumption, main = "Q-Q Plot for Energy Consumption")
qqline(aggregated_df$energy_consumption, col = "blue")

# Perform ANOVA or Kruskal-Wallis Test for Energy Consumption
task_groups <- split(aggregated_df$energy_consumption, aggregated_df$Task)

if (length(task_groups) > 1) {
  anova_result <- aov(energy_consumption ~ Task, data = aggregated_df)
  cat("ANOVA Result:\n")
  print(summary(anova_result))
  
  if (summary(anova_result)[[1]]$`Pr(>F)`[1] < 0.05) {
    tukey_result <- TukeyHSD(anova_result)
    cat("Tukey's HSD Post-hoc Test:\n")
    print(tukey_result)
  }
} else {
  kruskal_result <- kruskal.test(energy_consumption ~ Task, data = aggregated_df)
  cat("Kruskal-Wallis Test Result:\n")
  print(kruskal_result)
}

# Correlation Analysis
pearson_corr <- cor.test(aggregated_df$`Duration (s)`, aggregated_df$energy_consumption, method = "pearson")
cat("Pearson Correlation:\n")
print(pearson_corr)

spearman_corr <- cor.test(aggregated_df$`Duration (s)`, aggregated_df$energy_consumption, method = "spearman")
cat("Spearman Correlation:\n")
print(spearman_corr)

# Resource Utilization Analysis: GPU and CPU
gpu_groups <- split(aggregated_df$`GPU Utilization (%)`, aggregated_df$Task)
cpu_groups <- split(aggregated_df$`CPU Utilization (%)`, aggregated_df$Task)

if (length(gpu_groups) > 1) {
  gpu_anova_result <- aov(`GPU Utilization (%)` ~ Task, data = aggregated_df)
  cat("GPU ANOVA Result:\n")
  print(summary(gpu_anova_result))
  
  if (summary(gpu_anova_result)[[1]]$`Pr(>F)`[1] < 0.05) {
    gpu_tukey_result <- TukeyHSD(gpu_anova_result)
    cat("Tukey's HSD for GPU Utilization:\n")
    print(gpu_tukey_result)
  }
}

if (length(cpu_groups) > 1) {
  cpu_anova_result <- aov(`CPU Utilization (%)` ~ Task, data = aggregated_df)
  cat("CPU ANOVA Result:\n")
  print(summary(cpu_anova_result))
  
  if (summary(cpu_anova_result)[[1]]$`Pr(>F)`[1] < 0.05) {
    cpu_tukey_result <- TukeyHSD(cpu_anova_result)
    cat("Tukey's HSD for CPU Utilization:\n")
    print(cpu_tukey_result)
  }
}

# Box Plot for Energy Consumption by Task
ggplot(aggregated_df, aes(x = Task, y = energy_consumption)) +
  geom_boxplot(fill = "steelblue", color = "black") +
  ggtitle("Energy Consumption by Task") +
  xlab("Task") +
  ylab("Energy Consumption (J)") +
  theme_minimal()

# Summary of Results
cat("\n--- Statistical Test Summary ---\n")
cat("Shapiro-Wilk Test p-value:", shapiro_test$p.value, "\n")
if (exists("anova_result")) {
  cat("ANOVA p-value:", summary(anova_result)[[1]]$`Pr(>F)`[1], "\n")
} else if (exists("kruskal_result")) {
  cat("Kruskal-Wallis p-value:", kruskal_result$p.value, "\n")
}
