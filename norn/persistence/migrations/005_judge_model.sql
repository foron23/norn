-- NOR-19: judge ensemble multi-modelo — cada judge registra su llamada con
-- role='judge' + modelo en metadata para que estimate_campaign_cost pueda
-- distinguir precios por modelo de judge (no solo el modelo de campaña).
ALTER TABLE turn_event ADD COLUMN model TEXT;
