# 🏗️ Evaluación de Arquitectura V3: Edge-Native Agentic Platform

> **Objetivo Estratégico:** Transformar Autopilot de una plataforma de "pipelines secuenciales" (V2) a un **sistema agéntico de clase mundial** (V3) alineado con los patrones avanzados del Google ADK (Agent Development Kit).

Esta guía está diseñada para que cualquier desarrollador (junior o senior) pueda entender **qué** hay que hacer, **por qué** y **cómo** implementarlo paso a paso.

## 📊 Summary & Progress Tracker

| Fase  | Componente                      | Descripción                                                                     | Complejidad |    Estado     |
| :---: | :------------------------------ | :------------------------------------------------------------------------------ | :---------: | :-----------: |
| **1** | **Native ADK Agents**           | Implementar adaptadores nativos para patrones ADK (Loop, Parallel, Sequential). |  🟡 Media   | ✅ Completado |
| **2** | **Multi-Strategy Orchestrator** | Soportar orquestación no lineal: DAGs, ReAct Loops y Routing dinámico.          |   🔴 Alta   | ✅ Completado |
| **3** | **Session & Memory Layer**      | Añadir memoria a largo plazo (Vector Store) y contexto de sesión persistente.   |   🔴 Alta   | ✅ Completado |
| **4** | **Tool Ecosystem**              | Crear un `ToolRegistry` centralizado y puentes para Connectors y MCPs.          |  🟡 Media   | ✅ Completado |
| **5** | **Agent Bus (A2A)**             | Bus de mensajes tipados para comunicación asíncrona entre agentes.              |   🔴 Alta   | ✅ Completado |
| **6** | **Declarative DSL**             | Definición de workflows complejos 100% en YAML.                                 |  🟡 Media   | ✅ Completado |

---

## 🧐 1. Evaluación Detallada del Estado Actual (V2)

El sistema actual (V2) es robusto y performante, pero limitado arquitectónicamente para casos de uso agénticos complejos.

### ✅ Fortalezas (Lo que mantenemos)

- **Core Primitives Sólidos**: `BaseAgent`, `Pipeline` y `AgentContext` proveen una base tipada y observable excelente.
- **Observabilidad World-Class**: Integración profunda con OpenTelemetry (Tracing) y SSE Streaming para real-time feedback.
- **Infraestructura de Plataforma**: `WorkflowRegistry` y `WorkflowRouter` manejan descubrimiento y routing de manera eficiente.
- **Connectors**: Abstracción limpia para integraciones externas (Gmail, YNAB).

### ❌ Debilidades Críticas (Gaps Arquitectónicos)

1.  **Orquestación Rígida (Secuencial O(N))**: Actualmente, `Pipeline` es estrictamente una lista lineal. No permite bucles de corrección ni ejecución paralela nativa.
2.  **Integración ADK Superficial**: `ADKAgent` es solo un wrapper básico. No aprovechamos `SequentialAgent`, `LoopAgent` ni `ParallelAgent` de Google ADK.
3.  **Ausencia Total de Memoria (Stateless)**: El sistema no tiene memoria entre ejecuciones.
4.  **Ecosistema de Herramientas Fragmentado**: No hay un registro central de herramientas reutilizables.

---

## 🚀 2. Visión V3: Arquitectura Edge-Native

La V3 introduce un **grafo de agentes autónomos** que comparten contexto y memoria, superando el modelo de pipeline lineal.

---

## 🛠️ 3. Roadmap de Implementación Detallado

A continuación, el plan de ejecución paso a paso con **instrucciones técnicas detalladas**.

### FASE 1: Native ADK Workflow Agents (Cimientos)

> **Goal:** Habilitar patrones de composición avanzados (bucles, paralelo) usando nativamente Google ADK.

#### 1.1. Crear `SequentialAgentAdapter`

- **Archivo:** `autopilot/core/agent.py`
- **Qué hacer:** Crear una clase `SequentialAgentAdapter` que herede de `BaseAgent`.
- **Detalle Técnico:** Esta clase debe envolver una instancia de `google.adk.agents.SequentialAgent`. En su método `run`, debe invocar al agente de ADK pasando el contexto de ejecución.
- **Por qué:** Para encadenar agentes de ADK (e.g. Prompt A -> Prompt B) dentro de un paso del pipeline de Autopilot, manteniendo la observabilidad.

#### 1.2. Crear `LoopAgentAdapter`

- **Archivo:** `autopilot/core/agent.py`
- **Qué hacer:** Crear una clase `LoopAgentAdapter` que herede de `BaseAgent`.
- **Detalle Técnico:** Wrapper para `google.adk.agents.LoopAgent`. Debe aceptar configuración de `max_iterations` y una función `exit_condition(state) -> bool`.
- **Code Snippet (Guía):**
  ```python
  class LoopAgentAdapter(BaseAgent):
      def __init__(self, agent: BaseAgent, condition: Callable[[dict], bool], max_iter: int = 3):
          # ... setup ...
      async def run(self, ctx, input):
          for i in range(self.max_iter):
              result = await self.agent.run(ctx, input)
              if self.condition(result): return result
          raise MaxRetriesExceededError()
  ```
- **Por qué:** Para permitir agentes que se auto-corrigen (e.g. "Generar JSON -> Validar -> Error -> Reintentar").

#### 1.3. Crear `ParallelAgentAdapter`

- **Archivo:** `autopilot/core/agent.py`
- **Qué hacer:** Crear wrapper para `google.adk.agents.ParallelAgent` o implementarlo con `asyncio.gather`.
- **Detalle Técnico:** Recibe una lista de `BaseAgent`. Ejecuta todos en paralelo. Espera a que todos terminen y fusiona sus resultados en un solo diccionario (o lista).
- **Por qué:** Para tareas como "Buscar en Google" Y "Buscar en Wikipedia" al mismo tiempo (Map-Reduce).

#### 1.4. Refactorizar `PipelineBuilder`

- **Archivo:** `autopilot/core/pipeline.py`
- **Qué hacer:** Añadir métodos fluent (`.loop()`, `.parallel()`) al builder.
- **Detalle Técnico:** Estos métodos instancian los adapters creados arriba y los añaden como pasos al pipeline.

**Definition of Done (Fase 1):**

- [x] Tests unitarios pasando para nuevos adapters.
- [x] Ejemplo de workflow que usa `.loop()` para reintentar una tarea fallida.

---

### FASE 2: Multi-Strategy Orchestration (Cerebro)

> **Goal:** Romper la linealidad del pipeline. Permitir grafos complejos y decisiones dinámicas.

#### 2.1. Definir `OrchestrationStrategy`

- **Archivo:** `autopilot/core/orchestrator.py` (nuevo)
- **Qué hacer:** Crear Enum `OrchestrationStrategy` con valores: `SEQUENTIAL`, `DAG`, `REACT`, `ROUTER`.

#### 2.2. Implementar `DAGBuilder`

- **Archivo:** `autopilot/core/dag.py`
- **Qué hacer:** Implementar lógica de grafos.
- **Detalle Técnico:**
  - Método `add_node(agent_name, agent, dependencies=['step_A'])`.
  - Al ejecutar, calcular el orden topológico (qué va primero, qué va después).
  - Ejecutar nodos sin dependencias en paralelo.
- **Por qué:** Para workflows complejos donde el paso D depende de B y C, pero B y C pueden correr en paralelo tras A.

#### 2.3. Actualizar `BaseWorkflow`

- **Archivo:** `autopilot/base_workflow.py`
- **Qué hacer:** Permitir configurar la estrategia. Si es `DAG`, usar el `DAGRunner` en lugar de `PipelineRunner`.

**Definition of Done (Fase 2):**

- [ ] Poder ejecutar un workflow definido como un grafo de dependencias.

---

### FASE 3: Session & Memory Layer (Contexto)

> **Goal:** Que el sistema recuerde lo que pasó ayer.

#### 3.1. Crear `SessionService`

- **Archivo:** `autopilot/core/session.py`
- **Qué hacer:** Gestionar estado a corto plazo (la conversación actual).
- **Detalle Técnico:** Interfaz `SessionService` con métodos `get(key)`, `set(key, value)`. Implementación en memoria para dev, y Redis opcional para prod.

#### 3.2. Crear `MemoryService` (Long-term)

- **Archivo:** `autopilot/core/memory.py`
- **Qué hacer:** Gestionar memoria semántica (Vector Database).
- **Detalle Técnico:**
  - Método `add_observation(text, metadata)`.
  - Método `search_relevant(query_text) -> list[Observation]`.
  - Usar una librería ligera (ej. ChromaDB o simple cosine similarity en memoria para empezar).
- **Por qué:** Para que un agente pueda preguntar "¿Cómo resolví este error la última vez?".

#### 3.3. Integrar en `AgentContext`

- **Archivo:** `autopilot/core/context.py`
- **Qué hacer:** Añadir `self.session` y `self.memory` al contexto que reciben todos los agentes.

**Definition of Done (Fase 3):**

- [ ] Un agente puede guardar un dato en memoria y otro agente puede recuperarlo en una ejecución futura.

---

### FASE 4: Tool Ecosystem (Capacidades)

> **Goal:** Reutilizar herramientas sin copiar código.

#### 4.1. Crear `ToolRegistry`

- **Archivo:** `autopilot/core/tools/registry.py`
- **Qué hacer:** Un diccionario global de herramientas.
- **Detalle Técnico:** Decorador `@tool` que registra una función y extrae su docstring y firma para el LLM.

#### 4.2. `Connector-as-Tool`

- **Archivo:** `autopilot/connectors/base_connector.py`
- **Qué hacer:** Que los conectores expongan métodos crudos.
- **Detalle Técnico:** Si tengo `YNABConnector`, quiero poder registrar `ynab.create_transaction` como tool automáticamente para que el LLM lo use si lo necesita.

#### 4.3. MCP Bridge

- **Archivo:** `autopilot/core/tools/mcp.py`
- **Qué hacer:** Cliente de Protocolo MCP.
- **Detalle Técnico:** Conectarse a un servidor MCP (ej. `brave-search`) y convertir sus herramientas en herramientas de Autopilot.

**Definition of Done (Fase 4):**

- [ ] Un agente LLM puede "ver" y usar herramientas registradas globalmente sin código extra en el workflow.

---

### FASE 5: Agent Bus (Adelantado A2A)

> **Goal:** Comunicación estilo "Slack" entre agentes.

#### 5.1. `AgentBus`

- **Archivo:** `autopilot/core/bus.py`
- **Qué hacer:** Sistema de mensajes.
- **Detalle Técnico:** Métodos `publish(topic, msg)` y `subscribe(topic, handler)`.
- **Por qué:** Para que un agente de "Monitoreo" pueda escuchar eventos de "Error" de cualquier otro agente y actuar proactivamente.

---

### FASE 6: Declarative DSL (Interface)

> **Goal:** Escribir workflows en YAML.

#### 6.1. Schema YAML

- **Archivo:** `workflow.yaml`
- **Qué hacer:** Definir sintaxis para todo lo anterior (steps, tools, memory, retry policies).

#### 6.2. `DSLLoader`

- **Archivo:** `autopilot/core/dsl_loader.py`
- **Qué hacer:** Leer el YAML e instanciar las clases Python correspondientes dinámicamente.

**Definition of Done (Fase 6):**

- [ ] Crear un workflow complejo funcional sin escribir ni una línea de Python (solo YAML y tools existentes).

---

> _Documento generado automáticamente por Antigravity AI Architect._
> _Última actualización: 2026-02-19_
