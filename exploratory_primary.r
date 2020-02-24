library(tidyverse)
library(lubridate)


df <- read_csv('data/2020_democratic_primary/pres_primary_avgs_2020.csv')

df <- df %>% 
	mutate(modeldate = lubridate::mdy(modeldate)) %>% 
	mutate(candidate_name = ifelse(candidate_name == 'JuliÃ¡n Castro', 'Julián Castro', candidate_name))

candidates <- df$candidate_name %>% unique()
dates <- df$modeldate %>% unique()

averages <- df %>% 
	filter(state == 'National') %>%
	group_by(candidate_name, modeldate) %>% 
	summarise(mean_pct = mean(pct_estimate)) %>% 
	ungroup() %>% 
	group_by(candidate_name) %>% 
	mutate(entrance = min(modeldate, na.rm = T), 
				 exit = max(modeldate, na.rm = T)) %>% 
	ungroup() %>% 
	tidyr::complete(modeldate, nesting(candidate_name, entrance, exit)) %>% 
	group_by(candidate_name) %>% 
	arrange(modeldate) %>%
	tidyr::fill(mean_pct, .direction = 'down') %>% 
	mutate(mean_pct = ifelse(modeldate < entrance | modeldate > exit,
													 0,
													 mean_pct))
	
	
averages %>% 
	ggplot() + 
	aes(x = modeldate, y = mean_pct, color = candidate_name, group = candidate_name) + 
	geom_line() + 
	theme_classic() 

ggsave('fig/2020_primary_exploratory.png', dpi = 300, height = 4, width = 10)

# not sure if we need to construct this data frame or not. 
pairwise_df <- expand.grid(candidate1 = candidates, 
						candidate2 = candidates, 
						date = dates) %>% 
	tbl_df() %>% 
	left_join(averages, by = c('candidate1' = 'candidate_name', 
														 'date' = 'modeldate')) %>% 
	left_join(averages, by = c('candidate2' = 'candidate_name', 
														 'date' = 'modeldate'),
						suffix = c('1','2')) %>% 
	replace_na(list('mean_pct1' = 0, 'mean_pct2' = 0))
	
averages %>% write_csv('data/2020_democratic_primary/2020_primary_national_averages.csv')

