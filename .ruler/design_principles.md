When designing the codebase and implementing function. **Always** follow ArjanCodes principles:

# A Definitive Synthesis of ArjanCodes' Python Software Design Principles

## Part 1: The Core "ArjanCodes" Design Philosophy

### Section 1.1: The Guiding Mindset: Pragmatism Over Dogma

The central philosophy of the ArjanCodes channel is the cultivation of a "Software Designer Mindset".1 The primary objective of this mindset is not to demonstrate technical cleverness or to dogmatically apply theoretical patterns, but to consistently produce software that is robust, maintainable, and "easy to work on".2 This philosophy is built on a foundation of pragmatism, where principles and patterns are viewed as tools to be applied judiciously, rather than as rigid rules.

The "Problem First" Mandate

The most critical phase of any design process is the initial "understanding [of] the problem you're trying to solve".2 This is identified as a step that a significant number of developers prematurely skip, leading to poorly-defined solutions.2 Before any code is written or any pattern is chosen, a deep and thorough analysis of the problem, its context, and its requirements is mandatory.

The Foundational Triad (YAGNI, DRY, KISS)

Three core software principles form the bedrock of this pragmatic approach 3:

1. **YAGNI (You Ain't Gonna Need It):** This principle is the primary defense against overengineering.3 It mandates that developers should "not write unnecessary code" 5 and should only implement functionality that is required *now*, not what is imagined to be needed in the future. Complexity should only be added "when you have no other choice".5
2. **DRY (Don't Repeat Yourself):** This principle is focused on reducing the repetition of code and logic.3 Duplication is a primary source of software fragility and maintenance overhead, and it is the direct antagonist to the "duplicate code" code smell.6
3. **KISS (Keep It Simple):** This philosophy posits that "simplicity is robust, complexity is fragile" 5 and "simplicity is the ultimate sophistication".4 This mindset is the justification for many specific refactoring techniques, such as avoiding deep nesting 7 or complex boolean expressions.8

The 7-Step Design Framework

To move from a problem to a solution, a 7-step design framework is proposed, providing a structured process for applying the designer mindset.2 This framework ensures all critical aspects of the design are considered 2:

1. **Define What You're Building:** Clearly articulate the problem, the intended audience, and the main concepts involved.2
2. **Design the User Experience:** Define the main user stories, "happy flows," alternative flows, and any UI mockups.2
3. **Understand the Technical Needs:** Detail the technical specifications, including database schemas, algorithms, required libraries, and the design patterns used to model the concepts.2
4. **Implement Testing and Security Measures:** Define coverage goals, types of tests (unit, regression, etc.), and necessary security checks or audits.2
5. **Plan the Work:** Create time estimates, define developmental milestones, identify risk factors, and establish a clear "Definition of Done".2
6. **Identify Ripple Effects:** Consider tasks beyond implementation, such as updating documentation, communicating with users, and updating external systems (e.g., payment providers).2
7. **Understand the Broader Context:** Identify the limitations of the current design and potential future extensions.2

The Strategic Value of "Moonshot" Ideas

The final step of the design framework, "Understand the Broader Context," includes the recommendation to add "moonshot" ideas, such as, "It would be really cool if the software could also do X".2 At first, this appears to be in direct contradiction to the YAGNI principle.3

However, the purpose of this exercise is not *implementation* but *ideation*. Its value is threefold:

1. It "puts you in a very open mindset".2
2. It allows a developer to recognize "epiphanies" during development that might make the moonshot idea feasible.2
3. It serves as a "starting point" for future improvements.2

This "moonshot" concept is a pragmatic tool for informing the design of abstractions. By considering a potential (but not implemented) future extension, the developer can make more informed decisions about the system's "seams" or abstraction boundaries. This directly supports the Open-Closed Principle. By thinking about a "moonshot," the developer can design the *current* system to be "open" to this future extension without requiring "modification" of the existing core, thus creating a more resilient and future-proof (though still simple) design.

### Section 1.2: The Pillars of Good Design: High Cohesion and Low Coupling

While YAGNI, DRY, and KISS are the guiding philosophies, the *primary technical metrics* for evaluating design quality are Cohesion and Coupling.9 These concepts are the foundation of the "Write BETTER PYTHON CODE" series, which presents them as part of the GRASP (General Responsibility Assignment Software Patterns) principles.9 Design patterns and principles like SOLID are ultimately tools to achieve the goal of high cohesion and low coupling.

High Cohesion (Do One Thing)

Cohesion refers to the degree to which the elements inside a single module, class, or function belong together.9 The goal is to achieve High Cohesion.

In practice, this translates directly to the Single Responsibility Principle, but applied at all levels. A function should "Do One Thing".3 A module should be responsible for one "thing" or "feature".10 When a function or module attempts to do too many unrelated things, it has low cohesion and becomes difficult to understand, test, and maintain.

Low Coupling (Minimize Dependencies)

Coupling is the measure of how dependent one module is on the internal implementation details of another module.9 The goal is to achieve Low Coupling, where modules interact through stable, abstract interfaces.

The "Write BETTER PYTHON CODE" series is a multi-part demonstration of how to achieve low coupling:

- Part 2 introduces the Dependency Inversion Principle.9
- Part 3 introduces the Strategy Pattern.9
- Part 4 introduces the Observer Pattern.9

These are all presented as specific techniques for reducing coupling and managing dependencies effectively.11

The pedagogical structure of this content is significant. Cohesion and Coupling are introduced in Part 1 9, while the full SOLID principles are not detailed until Part 9.9 This implies a conceptual hierarchy: Cohesion and Coupling are the *primary, fundamental metrics* for design quality. SOLID principles and design patterns are a *collection of tools and heuristics* used to achieve this primary goal.

Therefore, any proposed code change should first be evaluated through this lens:

1. Does this change **increase cohesion** by grouping related logic?
2. Does this change **decrease coupling** by depending on abstractions rather than concretions?

## Part 2: A Pragmatic Guide to Software Architecture

### Section 2.1: The Anatomy of a Scalable Python Project

A recurring theme is the creation of a practical, scalable "blueprint" for a modern Python application, as demonstrated in the "Anatomy of a Scalable Python Project (FastAPI)" video.12

Definition of "Scalable": Balanced, Not Bloated

The first principle of this architecture is "Balanced, Not Bloated".12 This is a direct rejection of overengineering.3 The goal is not to start with a complex, enterprise-grade framework, but to establish a clean and simple structure that can scale as requirements evolve.

Folder Structure and Configuration

The blueprint for this balanced project includes:

- **Folder Structure:** A clean folder structure, including a `tests/` directory that is configured to mirror the source directory's structure for discoverability and maintainability.12
- **Configuration Management:**
    - `pyproject.toml`: The modern standard for defining project metadata and managing dependencies.12
    - `.python-version`: To ensure consistent Python environments across development and production.12
    - `.env` files: For "centralized, boring, safe" configuration management, adhering to the tip to "move configuration to a separate file".12
- **Environment and Tooling:**
    - `Docker` and `docker-compose`: Used to create "consistent environments and easy onboarding" for all developers.12

Essential Services: Logging and Dependency Injection

A scalable project formalizes its core services:

- **Centralized Logging:** Logging is configured in "one door in".12 Individual modules do not configure logging; they simply request a logger. This ensures consistent logging formats and outputs.
- **Dependency Injection (DI):** The architecture relies heavily on DI, leveraging framework "FastAPI Built-Ins" rather than adding a heavier, external DI framework.12

The "Business Seam": Pragmatic Clean Architecture

This architecture's most powerful concept is the "UserService as the 'Business Seam'".12 This "seam" is a pragmatic implementation of Clean Architecture principles.

1. **The Problem:** In many web applications, business logic becomes tightly coupled to the web framework (e.g., in the API route handlers). This makes the logic hard to test and impossible to reuse.
2. **The Solution:** The "Business Seam" is an abstraction boundary. All core domain logic (e.g., how to create, validate, or find a user) is encapsulated within a "Service" class (e.g., `UserService`).
3. **The Connection:** The web framework's API routes (e.g., the FastAPI endpoints) are forbidden from containing this logic. Instead, they *depend on* the `UserService`.
4. **The Mechanism:** The framework's built-in Dependency Injection system is used to *inject* an instance of the `UserService` into the route handler that needs it.12

This pattern is a direct, practical application of the **Dependency Inversion Principle (DIP)** at an architectural level. The web layer (a high-level detail) and the database persistence layer (a low-level detail) are both decoupled from the core business rules. Both depend on the abstraction (the `UserService` seam).

The result is a system that is highly maintainable and testable. The `UserService` and its business logic can be tested "fast and hermetic" 12, completely isolated from the web framework or database, simply by instantiating it.

### Section 2.2: A Pragmatic Re-evaluation of SOLID Principles in Python

The SOLID principles are frequently discussed, but not as immutable laws. They are presented as valuable heuristics that must be adapted to the Pythonic context, as explored in "SOLID: Writing Better Python Without Overengineering".17

The core takeaway is that the principles themselves are useful, but their "classic" Java-centric, class-based *implementations* are often not the most "Pythonic" or simple solution.18 The *principle* must be separated from the *implementation*.

**S - Single Responsibility Principle (SRP)**

- **Principle:** "A class should have one and only one reason to change".19
- **Pythonic Take:** This principle is even more important for *functions*. The primary rule for functions is "Functions/Modules Should Be Responsible For One Thing".10
- **Example:** In a payment system example, an initial `Order` class handles both order items and payment logic.20 This violates SRP. The "fix" is to separate the payment logic into a new `PaymentHandler` class, giving each class a single responsibility.20

**O - Open/Closed Principle (OCP)**

- **Principle:** "Software entities... should be open for extension but closed for modification".20
- **Pythonic Take:** This is the primary tool for eliminating long `if-elif` chains. When new behavior is needed, it should be possible by *adding* new code, not *changing* old code.
- **Example:** The new `PaymentHandler` class has methods like `pay_debit` and `pay_credit`.20 To add PayPal, the class must be *modified* (violating OCP). The "fix" is to use an abstract base class, `PaymentHandler(ABC)`, and create concrete subclasses (`DebitPaymentHandler`, `CreditPaymentHandler`). To add PayPal, one simply *adds* a new `PayPalPaymentHandler` class (extension) without touching the others.20 This is the core logic of the Strategy Pattern.

**L - Liskov Substitution Principle (LSP)**

- **Principle:** "Objects of a superclass should be replaceable with objects of its subclasses without breaking the program".20
- **Pythonic Take:** This serves as a *validation test* for abstractions. If the code must use `isinstance()` to check a subclass's type before calling a method, the LSP is being violated, and the abstraction is flawed.6
- **Example:** Following the OCP fix, a problem arises: `DebitPaymentHandler` requires a `security_code`, but `PayPalPaymentHandler` requires an `email`.20 If the abstract `pay` method signature is modified to accommodate both, it breaks the contract for all subclasses. The LSP-compliant "fix" is to move the varying data (the `security_code` or `email`) into the *constructor* (`__init__`) of the *concrete* subclasses. This allows the `pay(order)` method signature to remain identical and substitutable across all payment handlers.20

**I - Interface Segregation Principle (ISP)**

- **Principle:** "Clients should not be forced to depend on interfaces they do not use".4
- **Pythonic Take:** This principle is a powerful argument for **Composition over Inheritance**.4
- **Example:** To add two-factor authentication (2FA), a naive approach would add an abstract `auth_2fa_sms` method to the base `PaymentHandler` interface.20 This forces a `CreditPaymentHandler` (which doesn't support 2FA) to implement a method it doesn't use, violating ISP. The "fix" is to create a *separate, smaller interface* (e.g., `Authorizer(ABC)`) and *inject* (compose) a concrete `SMSAuthorizer` instance *only* into the payment handlers that require it.20

**D - Dependency Inversion Principle (DIP)**

- **Principle:** "High-level modules should not depend on low-level modules. Both should depend on abstraction".4
- **Pythonic Take:** This is the most important *architectural* principle. It is the enabling concept behind the "Business Seam" 12, the reason to "Use dependency injection" 21, and the key to testability.
- **Example:** The ISP fix is completed by DIP. The `DebitPaymentHandler` (high-level module) does *not* depend on the concrete `SMSAuthorizer` (low-level module). Instead, it depends on the abstract `Authorizer(ABC)` *interface*. This *inverts* the dependency, decoupling the modules and allowing a different authorizer (e.g., `reCAPTCHA_Authorizer`) to be injected without any change to the payment handler.20

## Part 3: Implementation Guide: Patterns, Classes, and Functions

### Section 3.1: Strategic Application of Design Patterns

Design patterns are treated as solutions to specific, recurring problems, not as items on a checklist. The emphasis is on recognizing the *problem* (often a "code smell") and applying the appropriate pattern to solve it.

- **Problem:** Long, complex `if-elif` chains.13
    - **Solution:** **Strategy Pattern**.9
    - **Implementation:** This pattern allows for pluggable behaviors. This can be implemented in a "classic" class-based way (using an abstract base class, as in the OCP example 20) or in a more lightweight, Pythonic way using a *dictionary* that maps keys to *functions*.23 For example, "Personalities as Pluggable Agent Prompts".22
- **Problem:** Complex object creation logic is scattered and coupled to the client code.
    - **Solution:** **Factory Pattern**.24
    - **Implementation:** The `before.py` file in the example repository shows creation logic (e.g., `if-elif` blocks) in the `main` function. The `after.py` file refactors this by creating a `Factory` class that encapsulates this logic, separating the *creation* of an object from its *use*.24
- **Problem:** A linear process needs to be built from modular, re-chainable steps.
    - **Solution:** **Chain of Responsibility Pattern**.22
    - **Implementation:** Demonstrated in the context of building an "AI Travel Pipeline".22 Each processing step is a "handler" that can either process the request or pass it to the next handler in the chain.
- **Problem:** Multiple, unrelated objects need to be notified of a state change in one object.
    - **Solution:** **Observer Pattern**.9
    - **Implementation:** Used for "Logging Agent Behavior with Context".22 A "publisher" object maintains a list of "subscriber" objects. When the publisher's state changes, it iterates through its subscribers and notifies them. This can also be implemented functionally using a list of callbacks.23
- **Problem:** An algorithm's high-level structure is fixed, but the low-level implementation of its steps must be interchangeable.
    - **Solution:** **Template Method Pattern** 25 and **Bridge Pattern**.25
    - **Implementation:** The Template Method defines the algorithm's skeleton in a base class.25 Subclasses can override specific steps. The Bridge Pattern is presented as a more flexible, composition-based alternative to simple inheritance, where the implementation *object* is *injected* into the main class.25

The following table summarizes the application of these patterns as pragmatic solutions to common problems.

**Table 1: Pragmatic Design Pattern Application**

| **Problem (Code Smell)** | **Pattern** | **Classic (Class-Based) Implementation** | **Pythonic (Functional) Alternative** |
| --- | --- | --- | --- |
| Long `if-elif` chain; need for pluggable behaviors.13 | **Strategy** 22 | An abstract `Strategy` interface. Concrete classes `ConcreteStrategyA`, `ConcreteStrategyB`. | A dictionary mapping string keys to functions: 
 `strategies = {"a": do_a_func, "b": do_b_func}`. |
| Multiple objects need to be notified of state changes. | **Observer** 22 | `Publisher` class with `attach()`/`notify()` methods. `Subscriber` interface with `update()`. | `Publisher` class maintains a list of *callback functions* and calls them on state change. |
| Complex object creation logic is coupled to the client. | **Factory** 24 | `Creator` abstract class with a `factory_method`. Concrete creators return concrete products.24 | A single function that takes a type parameter (e.g., a string or `Enum`) and returns the correct object. |
| A pipeline of processing steps is needed. | **Chain of Responsibility** 22 | `Handler` interface with `set_next()` and `handle()` methods. `handle()` either processes or passes to `next`. | A list or sequence of functions. A `pipeline` function iterates through the list, passing the result of one to the next. |
| Algorithm skeleton is fixed, but steps are variable. | **Template Method** 25 | An abstract base class defines the main `template_method()`, which calls abstract "hook" methods that subclasses must implement. | A high-level function that accepts other *functions* as arguments for the variable steps (akin to the Bridge pattern). |

### Section 3.2: The Ultimate Guide to Writing Classes and Functions

The choice of *which* programming construct to use follows a clear heuristic that prioritizes simplicity.

The Heuristic: Function -> Dataclass -> Class

A clear preference hierarchy emerges from the content:

1. **Default to Functions:** A standalone function is the simplest, most testable, and most "Pythonic" unit of logic. The default should always be a function.3
2. **Use `dataclass` for State:** When simple state (data) needs to be passed around, the first choice should be a `dataclass`. This provides "Less Boilerplate, More Clarity" 27 and is superior to a plain class with just an `__init__` or a `tuple` or `dict`.
3. **Use `class` for State + Behavior:** A full `class` should *only* be introduced when there is a clear need to co-locate and encapsulate *both* state (data members) and the *behavior* (methods) that operates on that state.

This is a critical "guard rail" against overengineering. The "Ultimate Guide to Writing Classes in Python" 21 spends a significant portion of its time on the topic "Make sure a class is actually needed".21

Rules for Effective Functions

When using functions, they must be "clean":

- **DO:** Keep them short and ensure they "Do One Thing" (High Cohesion).3
- **DO:** Use "Meaningful Names".3 A "vague identifier" is a code smell.6
- **DO:** "Comment Wisely".10 This means "Document thought process, not just what the code does".3 Document the *why* (the design decision) not the *what* (e.g., "this loop increments i").
- **DON'T:** Pass "Too many parameters".7 This is a code smell. Group them into a `dataclass`.27
- **DON'T:** Use "boolean flags" to make a function do two different things.6 This is a clear violation of SRP. Create two separate functions instead.
- **DON'T:** Treat "Error Handling as Logic".10 Use exceptions for exceptional, non-recoverable cases.

Rules for Effective Classes

When a class is truly necessary:

- **DO:** Keep your classes small.21 A large class is a sign of low cohesion.
- **DO:** Use "Encapsulation".21 Hide internal state and expose behavior.
- **DO:** "Use dependency injection".21 A class should *not* create its dependencies; it should *receive* them (in the `__init__`).20 This is the key to testability and low coupling (DIP).
- **DON'T:** Use `self` when it's not needed.7 If a method does not access `self`, it should be a `@staticmethod` or, more likely, a standalone function in the module.

## Part 4: The Developer's Safety Net: A Methodology for Testing

Testing is not an afterthought; it is a core component of the design process.2 A robust test suite is the "safety net" that allows for confident refactoring and maintenance. The "10 Tips to Keep Your Software Simple" includes "Write Tests for Critical Code".3

### Section 4.1: The Pytest-centric Testing Framework

The recommended testing methodology is "pytest-centric" and emphasizes simplicity and minimalism, as demonstrated in "This Is How Marie Kondo Sets up Her Pytest".14

**The "Marie Kondo" Setup**

- **Tool:** Use `pytest` as the test runner. It should be installed as a development dependency.14
- **Structure:** Create a top-level `tests/` folder. "Use the same folder structure in your test folder as for your source folder".14 This ensures that `tests/test_services.py` tests the code in `src/services.py`, making tests easy to locate.
- **Configuration:** Configure the IDE (e.g., VS Code) for `pytest` test discovery and running.14

Writing "Hermetic" Tests

The goal is to write tests that are "Fast and Hermetic".12 A hermetic test is fully self-contained, isolated, and deterministic.

- **Rule 1: One Concept per Test:** "make sure you have one assert per function".14 This keeps the test focused on a single behavior, and a failure immediately pinpoints the problem.
- **Rule 2: Independent Tests:** "make sure you reset state before each test so that one test is not dependent on another test".14 Tests should be ableS to run in any order and should never rely on the side effects of a previous test.
- **Rule 3: Good Names:** "make sure your test names make sense" 14 (e.g., `test_payment_processor_fails_on_expired_card`).

Effective pytest Features

To write clean, DRY tests, use pytest's built-in features:

- **Fixtures:** "use fixtures to reduce repetition".14 A fixture is a function that provides a resource (like a database connection, a sample object, or a mock) to a test.30 This is superior to `setUp/tearDown` methods and avoids duplicating object creation in every test.
- **Parametrization:** This is one of the "Neat Pytest Features".30 It allows a single test function to be run with multiple different sets of inputs, which is ideal for testing edge cases (e.g., `None`, empty strings, zero, negative numbers) without writing duplicate test code.

### Section 4.2: Isolating Code: Mocking vs. Monkey Patching

Testing code with external dependencies (e.g., APIs, databases) requires isolation. The terminology is clarified: **Patching** is the *action* of replacing a dependency. The **Mock** (e.g., `unittest.mock.MagicMock`) is the *test-double* you replace it with. **Monkey Patching** is `pytest`'s specific feature for performing this patching.30

"Refactor for Testability" is Superior to Patching

The most critical concept in testing is that patching is often a design smell. Videos on adding tests to existing code 32 and the "not ideal" setup with a "real API key" 32 highlight that it is difficult to test poorly designed code. The "How to Write Great Unit Tests" video has a dedicated chapter: "Refactor for Testability".30

This leads to a clear, actionable heuristic for the AI:

1. **The Problem:** A developer has a function with a hard-coded dependency, making it hard to test:Python
    
    # 
    
    `def get_weather_data():
        # Hard-coded dependency on a global module
        response = requests.get("https://api.weather.com/...")
        return response.json()`
    
2. **The "Bad" Solution (Patching):** The developer uses `pytest-monkeypatch` to replace the `requests.get` function *globally*. This is brittle (it's a global change), complex, and couples the test to the *implementation detail* that `requests.get` is used.
3. **The "Good" Solution (Refactor for Testability):** The developer applies the **Dependency Inversion Principle (DIP)**.
    - **Step A (Refactor):** Create an abstraction and use dependency injection.Python
        
        # 
        
        `class WeatherClient(ABC):
            @abstractmethod
            def get(self) -> dict:...
        
        def get_weather_data(client: WeatherClient) -> dict:
            # Function now depends on the abstraction
            return client.get()`
        
    - **Step B (Test):** The test is now trivial, clean, and fast. No patching is required.Python
        
        # 
        
        `def test_get_weather_data():
            # Create a mock object that respects the interface
            mock_client = MagicMock(spec=WeatherClient)
            mock_client.get.return_value = {"temp": 25}
        
            # Inject the mock
            assert get_weather_data(mock_client) == {"temp": 25}`
        

This approach, "Refactoring for Testability," is always preferred over patching. Patching should be a last resort. The need to patch often indicates a design violation, specifically a lack of Dependency Inversion.

## Part 5: Master "Dos and Don'ts": A Canonical List for the AI Prompt

This final section synthesizes the entire philosophy into a definitive, actionable set of rules, perfectly suited for a high-fidelity AI system prompt.

### Section 5.1: The "Don'ts": A Taxonomy of Code Smells

This is a comprehensive "don't" list, representing low-quality code that must be refactored.

**Table 2: Code Smell Identification and Refactoring Guide**

| **Code Smell** | **Description (The "Bad" Code)** | **The "Good" Refactor (Arjan's Solution)** | **Source** |
| --- | --- | --- | --- |
| **Imprecise Types** | `def process_items(items: list)` | Use precise types. `def process_items(items: list[str])`. Use `mypy`.33 | 6 |
| **Duplicate Code** | Copy-pasted logic in multiple places. | Abstract the common logic into a single function or class (DRY).3 | 6 |
| **Not Using Built-ins** | A manual `for` loop to find a max value. | Use built-in functions like `max()`, `any()`, `all()`, or `itertools`.27 | 6 |
| **Vague Identifiers** | `def process_data(d, l)` | Use meaningful, specific names: `def validate_user_profile(profile: UserProfile)`.3 | 6 |
| **`isinstance` for Behavior** | `if isinstance(x, A): x.do_a() elif isinstance(x, B): x.do_b()` | Use Polymorphism (LSP) or Strategy Pattern. Define a common `x.do()` interface. | 6 |
| **Boolean Flags** | `def save_user(user, send_email=False)` | Create two separate, clear functions: `save_user(user)` and `save_user_and_send_email(user)`. | 6 |
| **Ignoring Exceptions** | `try:... except: pass` | Catch *specific* exceptions (e.g., `except FileNotFoundError:`) and handle them, or let it propagate. | 6 |
| **Not Using Custom Exceptions** | `raise Exception("Payment failed")` | Define and raise a custom, specific exception: `class PaymentError(Exception): pass`... `raise PaymentError(...)`. | 6 |
| **Too Many Parameters** | `def create_user(name, email, age, address, phone)` | Group parameters into a `dataclass` 27: `def create_user(user_data: UserData)`. | 7 |
| **Too Deep Nesting** | `if... for... if... if...` | Use "guard clauses" (early returns) or refactor the inner logic into a new, smaller function. | 7 |
| **Wrong Data Structure** | Using a `list` (O(n) lookup) when you need fast lookups. | Use the correct data structure: a `set` or `dict` for O(1) lookups. | 7 |
| **Wildcard Imports** | `from my_module import *` | Be explicit. `from my_module import specific_function, specific_class`. | 7 |
| **Asymmetrical Code** | `f = open(...)` without a `f.close()` in all code paths. | Use context managers for resource management: `with open(...) as f:`.28 | 7 |
| **`self` Not Used** | `class MyClass: def my_func(self): return "hello"` | The method doesn't use instance state. Make it a `@staticmethod` or (better) a module-level function. | 7 |
| **Not Using `main()`** | Script-level code (e.g., `run_app()`) at the top level of a file. | Put script logic into a `main()` function and call it via `if __name__ == "__main__": main()`. | 7 |
| **Hardcoded Values** | `if payment_status == 1:...` | Use constants or an `Enum`: `if payment_status == PaymentStatus.PAID:...`.3 | 3 |

### Section 5.2: The "Dos": A Checklist for Clean, Pythonic Code

This is the positive, "do this" checklist that encapsulates the entire "Software Designer Mindset."

- **DO** always start by applying the **7-Step Design Framework** 2 to fully understand the problem, context, and requirements *before* writing any code.
- **DO** evaluate every design decision against the primary metrics of **High Cohesion** (grouping related logic) and **Low Coupling** (depending on abstractions).9
- **DO** write code that is "Balanced, Not Bloated".12 Adhere to **YAGNI**, **DRY**, and **KISS**.3
- **DO** follow the **"Function -> Dataclass -> Class"** heuristic. Default to functions; use `dataclasses` 27 for data; use `class` only when co-locating state and behavior.21
- **DO** leverage Python's modern standard library:
    - **`dataclasses`** for data-centric objects.27
    - **`pathlib`** for all file system path manipulation.27
    - **`functools`** (e.g., `cached_property`, `cache`) for optimization.27
    - **`itertools`** for complex, efficient iteration.27
    - **`tomllib`** for parsing TOML configuration files.27
- **DO** separate core **domain logic** from the framework (e.g., FastAPI, Flask).
- **DO** encapsulate this domain logic in "Service" classes (the "Business Seam").12
- **DO** use **Dependency Injection** (DIP) to provide these services to the framework, making the domain logic testable.12
- **DO** write fast, hermetic, and readable **`pytest` tests** for all critical code.3
- **DO** use `pytest` **Fixtures** to reduce test repetition and **Parametrization** to test edge cases.14
- **DO** *always* prefer **"Refactoring for Testability"** (using DIP) over using **Patching/Monkey-Patching**. The need to patch is a design smell.30
- **DO** structure **Data Science projects** with a common, standard layout 15, log results and metrics 15, and use intermediate data representations (e.g., Parquet, Feather) to decouple pipeline steps.15
- **DO** use the modern Python toolchain to enforce quality and consistency:
    - **`Black`** for automated, deterministic code formatting.33
    - **`mypy`** for static type checking (ideally in strict mode).33
    - **`Pylint`** or `Ruff` for linting.33
- **DO** document the **"why"** (the thought process and design decisions), not the **"what"** (the self-evident code).3